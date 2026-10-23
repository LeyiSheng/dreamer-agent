"""P1 supervised training on explicitly manifested frozen-visual exports.

No imagination, no automatic old checkpoint discovery, no real evaluation inputs.
Calibration metrics are diagnostic; this trainer does not select a best checkpoint.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import OfflineWindows
from .learning import HistoryPolicy, bc_loss
from .provenance import sha256, write_report


def run(manifest, output, *, epochs=1, context=32, targets=8, batch_size=8,
        hidden=128, lr=3e-4, weight_decay=0.01, seed=0, device="cpu", init_checkpoint=None):
    if epochs < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid training parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    train = OfflineWindows(manifest, "train", context, targets)
    calibration = OfflineWindows(manifest, "calibration", context, targets)
    m = train.manifest
    config = dict(feature_dim=m["feature_spec"]["dimension"], past_action_dim=sum(train.spec.action_sizes),
                  tasks=m["tasks"], action_sizes=train.spec.action_sizes, hidden=hidden, targets=targets,
                  task_conditioned=m.get("policy_mode", "task_conditioned") != "unconditioned")
    model = HistoryPolicy(**config).to(device)
    initialization = None
    if init_checkpoint is not None:
        prior = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        pm = prior["metadata"]
        expected = dict(config, task_conditioned=False, tasks=1)
        if json.loads(json.dumps(pm["config"])) != json.loads(json.dumps(expected)):
            raise ValueError("Initialization must be compatible unconditioned BC")
        if not config["task_conditioned"] or pm["feature_spec"] != m["feature_spec"] or pm["action_spec_sha256"] != train.spec.fingerprint:
            raise ValueError("Initialization feature/action contract mismatch")
        if pm["seed"] != seed or pm["manifest_sha256"] != m.get("parent_manifest_sha256"):
            raise ValueError("Initialization seed/source manifest mismatch")
        state = prior["model"]
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite initialization")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing != ["task.weight"] or unexpected:
            raise ValueError("Unexpected initialization keys")
        with torch.no_grad():
            model.task.weight.zero_()
        initialization = dict(path=str(Path(init_checkpoint).resolve()), sha256=sha256(init_checkpoint),
                              policy_mode=pm["policy_mode"], task_embedding="zero", optimizer="fresh")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    metadata = dict(schema_version=1, manifest_sha256=sha256(manifest), config=config,
                    action_spec_sha256=train.spec.fingerprint, feature_spec=m["feature_spec"],
                    plan_sha256=sha256(Path(__file__).resolve().parents[1] / "docs/PLAN_V6.md"),
                    code_sha256={name: sha256(Path(__file__).with_name(name))
                                 for name in ("learning.py", "dataset.py", "train_bc.py")},
                    seed=seed, epochs=epochs, context=context, batch_size=batch_size,
                    lr=lr, weight_decay=weight_decay, device=device, status="supervised_only",
                    policy_mode=m.get("policy_mode", "task_conditioned"))
    metadata["initialization"] = initialization
    write_report(output / "config.json", metadata)
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["features"], batch["past_actions"], batch["tasks"])
            loss = bc_loss(logits, batch["targets"], batch["valid"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            n = batch["valid"].sum().item()
            total += loss.item() * n
            count += n
        model.eval()
        cal_total, cal_count = 0., 0
        correct = [0] * len(train.spec.action_sizes)
        current_count = 0
        with torch.no_grad():
            for batch in DataLoader(calibration, batch_size=batch_size):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["features"], batch["past_actions"], batch["tasks"])
                n = batch["valid"].sum().item()
                cal_total += bc_loss(logits, batch["targets"], batch["valid"]).item() * n
                cal_count += n
                current_mask = batch["valid"][..., 0]
                current_count += current_mask.sum().item()
                for k, scores in enumerate(logits):
                    correct[k] += ((scores[..., 0, :].argmax(-1) == batch["targets"][..., 0, k]) & current_mask).sum().item()
        row = dict(epoch=epoch+1, train_nll=total/count, calibration_nll=cal_total/cal_count,
                   calibration_current_factor_accuracy=[c/current_count for c in correct],
                   current_targets=current_count, task_success_measured=False)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), metadata=metadata), output / "bc.pt")
    return output / "bc.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--init-checkpoint", type=Path)
    args = vars(p.parse_args())
    run(**args)


if __name__ == "__main__":
    main()
