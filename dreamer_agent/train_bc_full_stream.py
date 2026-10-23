"""Stream audited corpus sources through frozen tokenizer for unconditioned BC.

Supports --resume from latest.pt and multi-GPU source sharding
(--num-shards / --shard-id): each rank encodes/trains a disjoint subset of
remaining sources in parallel (decode-bound workload).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .export_vpt import PREPROCESS, decoded_frames, load_backend
from .learning import HistoryPolicy, bc_loss
from .provenance import sha256, write_report


def _windows(features, actions, *, context, targets, action_sizes):
  t = len(features)
  factors = len(action_sizes)
  past_dim = sum(action_sizes)
  out = []
  for start in range(0, t, context):
    stop = min(start + context, t)
    valid_length = stop - start
    feats = np.zeros((context, features.shape[1]), dtype=np.float32)
    past = np.zeros((context, past_dim), dtype=np.float32)
    tasks = np.zeros(context, dtype=np.int64)
    labels = np.zeros((context, targets, factors), dtype=np.int64)
    valid = np.zeros((context, targets), dtype=bool)
    feats[:valid_length] = features[start:stop]
    for i in range(valid_length):
      pos = start + i
      if pos > 0:
        offset = 0
        for k, n in enumerate(action_sizes):
          past[i, offset + int(actions[pos - 1, k])] = 1
          offset += n
      for horizon in range(targets):
        target = pos + horizon
        if target >= t:
          break
        labels[i, horizon] = actions[target]
        valid[i, horizon] = True
    if not valid.any():
      continue
    out.append(dict(features=feats, past_actions=past, tasks=tasks, targets=labels, valid=valid))
  return out


def _batches(windows, batch_size, rng):
  order = list(range(len(windows)))
  rng.shuffle(order)
  for i in range(0, len(order), batch_size):
    chunk = [windows[j] for j in order[i:i + batch_size]]
    yield {k: torch.from_numpy(np.stack([c[k] for c in chunk])) for k in chunk[0]}


def run(
  audit,
  output,
  *,
  seed=0,
  epochs=1,
  context=32,
  targets=8,
  batch_size=32,
  hidden=128,
  lr=3e-4,
  weight_decay=0.01,
  init_checkpoint=None,
  resume=None,
  device="cuda",
  num_shards=1,
  shard_id=0,
):
  if min(epochs, context, targets, batch_size, hidden, num_shards) < 1 or lr <= 0 or weight_decay < 0:
    raise ValueError("Invalid training parameters")
  if not (0 <= shard_id < num_shards):
    raise ValueError("shard_id out of range")

  audit = Path(audit)
  summary = json.loads((audit / "summary.json").read_text())
  if summary.get("status") != "audit_complete":
    raise ValueError("Audit incomplete")
  records = [json.loads(line) for line in (audit / "training_records.jsonl").read_text().splitlines()]
  eligible = [r for r in records if r.get("status") == "eligible_pending_full_decode"]
  train = [r for r in eligible if r["split"] == "train"]
  calibration = [r for r in eligible if r["split"] == "calibration"]
  if not train or not calibration:
    raise ValueError("Missing train/calibration sources")

  output = Path(output)
  if resume is not None:
    output.mkdir(parents=True, exist_ok=True)
  else:
    output.mkdir(parents=True, exist_ok=False)

  torch.set_num_threads(1)
  torch.manual_seed(seed + shard_id * 1009)
  spec, encode, action, extractor = load_backend(
    Path("outputs/acceptance_20260913/assets.json"),
    Path("outputs/acceptance_20260913/action_probe.json"),
    device)
  feature_spec = dict(
    kind="frozen_visual", frozen=True, dimension=8192,
    extractor_sha256=extractor,
    preprocessing_sha256=__import__("hashlib").sha256(
      json.dumps(PREPROCESS, sort_keys=True).encode()).hexdigest(),
    preprocessing=PREPROCESS)
  config = dict(
    feature_dim=8192, past_action_dim=sum(spec.action_sizes), tasks=1,
    action_sizes=spec.action_sizes, hidden=hidden, targets=targets,
    task_conditioned=False)
  model = HistoryPolicy(**config).to(device)

  start_source = 0
  updates = 0
  initialization = None
  resume_path = Path(resume) if resume is not None else None

  if resume_path is not None:
    prior = torch.load(resume_path, map_location="cpu", weights_only=False)
    pm = prior["metadata"]
    if pm["config"] != config or pm.get("policy_mode") != "unconditioned":
      raise ValueError("Resume checkpoint config mismatch")
    model.load_state_dict(prior["model"])
    start_source = int(prior.get("next_source_index", 0))
    updates = int(prior.get("updates", 0))
    initialization = dict(
      path=str(resume_path.resolve()),
      sha256=sha256(resume_path),
      note=f"resume next_source_index={start_source} updates={updates} shard={shard_id}/{num_shards}",
    )
  elif init_checkpoint is not None:
    prior = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
    pm = prior["metadata"]
    if pm["config"] != config or pm.get("policy_mode") != "unconditioned":
      raise ValueError("Init checkpoint must match unconditioned BC config")
    if pm["action_spec_sha256"] != spec.fingerprint:
      raise ValueError("Action spec mismatch vs init checkpoint")
    if pm["feature_spec"]["extractor_sha256"] != extractor:
      raise ValueError("Extractor mismatch vs init checkpoint")
    model.load_state_dict(prior["model"])
    initialization = dict(
      path=str(Path(init_checkpoint).resolve()), sha256=sha256(init_checkpoint),
      note="warm start from subset P1a; optimizer fresh")

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  if resume_path is not None and "optimizer" in prior:
    try:
      optimizer.load_state_dict(prior["optimizer"])
    except Exception as exc:
      print("OPT_RESUME_SKIP", type(exc).__name__, str(exc)[:120], flush=True)

  metadata = dict(
    schema_version=1, policy_mode="unconditioned", config=config,
    action_spec_sha256=spec.fingerprint, feature_spec=feature_spec,
    audit_sha256=sha256(audit / "training_records.jsonl"),
    audit_summary_sha256=sha256(audit / "summary.json"),
    plan_sha256=sha256(Path("docs/PLAN_V6.md")),
    code_sha256=sha256(__file__), seed=seed, epochs=epochs, context=context,
    targets=targets, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
    device=device, train_sources=len(train), calibration_sources=len(calibration),
    scope="all audited eligible train sources; one streaming pass per epoch; no corpus feature cache",
    status="supervised_only", initialization=initialization, rl_ready=False,
    num_shards=num_shards, shard_id=shard_id, resume_start_source=start_source)
  write_report(output / "config.json", metadata)

  def prepare(r):
    raw = Path(r["actions_path"])
    if sha256(raw) != r["actions_sha256"]:
      raise ValueError("Actions hash mismatch")
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    locals_ = np.stack([spec.to_local(np.asarray(action(row), dtype=np.int64)) for row in rows])
    vp = Path(r["video_path"])
    vh = sha256(vp)
    features = []
    for i, frame in enumerate(decoded_frames(vp)):
      if i < len(rows):
        features.append(encode(frame))
    if len(features) != len(rows):
      raise ValueError("Incomplete video decode")
    if sha256(raw) != r["actions_sha256"] or sha256(vp) != vh:
      raise ValueError("Source changed during encoding")
    return np.stack(features), locals_, vh

  failures = 0
  shard_seen = 0
  for epoch in range(epochs):
    order = list(train)
    random.Random(seed + 20260915 + epoch).shuffle(order)
    seen_video = set()
    train_total, train_count = 0.0, 0
    model.train()
    begin = start_source if epoch == 0 else 0
    for idx in range(begin, len(order)):
      if (idx - begin) % num_shards != shard_id:
        continue
      r = order[idx]
      try:
        features, actions, vh = prepare(r)
      except (ValueError, OSError) as exc:
        failures += 1
        with (output / "failed_sources.jsonl").open("a") as stream:
          stream.write(json.dumps(dict(
            epoch=epoch + 1, stem=r.get("stem"), shard_id=shard_id,
            error=f"{type(exc).__name__}: {exc}")) + "\n")
        print("SOURCE_FAILED", r.get("stem"), str(exc)[:150], flush=True)
        continue
      if vh in seen_video:
        continue
      seen_video.add(vh)
      windows = _windows(features, actions, context=context, targets=targets,
                         action_sizes=spec.action_sizes)
      source_updates = 0
      for batch in _batches(windows, batch_size, random.Random(seed + idx)):
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
        train_total += loss.item() * n
        train_count += n
        updates += 1
        source_updates += 1
      shard_seen += 1
      row = dict(
        epoch=epoch + 1, source_index=idx, stem=r.get("stem"), video_sha256=vh,
        updates=updates, source_updates=source_updates, shard_id=shard_id,
        mean_train_nll=(train_total / train_count) if train_count else None)
      with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
      torch.save(dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        metadata=metadata, next_source_index=idx + 1, updates=updates,
        epoch=epoch + 1, shard_id=shard_id, num_shards=num_shards,
      ), output / "latest.tmp")
      (output / "latest.tmp").replace(output / "latest.pt")
      print("TRAIN", f"shard{shard_id}", epoch + 1, idx + 1, len(order),
            r.get("stem"), updates, flush=True)

  if not updates and shard_seen == 0 and start_source >= len(train):
    print("NOTHING_LEFT", shard_id, flush=True)
  elif not updates and failures == 0 and shard_seen == 0:
    # This shard may have had zero assigned remaining sources.
    print("SHARD_EMPTY", shard_id, flush=True)
  if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
    raise RuntimeError("Nonfinite BC weights")
  metadata = dict(metadata, updates=updates, failed_train_sources=failures, shard_sources=shard_seen)
  torch.save(dict(model=model.state_dict(), metadata=metadata), output / "bc.pt")
  write_report(output / "summary.json", dict(
    status="complete", updates=updates, failed_train_sources=failures,
    policy_mode="unconditioned", shard_id=shard_id, num_shards=num_shards,
    shard_sources=shard_seen, resume_start_source=start_source))
  return output / "bc.pt"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--audit", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--epochs", type=int, default=1)
  p.add_argument("--context", type=int, default=32)
  p.add_argument("--targets", type=int, default=8)
  p.add_argument("--batch-size", type=int, default=32)
  p.add_argument("--hidden", type=int, default=128)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--weight-decay", type=float, default=0.01)
  p.add_argument("--init-checkpoint", type=Path, default=None)
  p.add_argument("--resume", type=Path, default=None, help="latest.pt with next_source_index")
  p.add_argument("--device", default="cuda")
  p.add_argument("--num-shards", type=int, default=1)
  p.add_argument("--shard-id", type=int, default=0)
  run(**{k.replace("-", "_"): v for k, v in vars(p.parse_args()).items()})


if __name__ == "__main__":
  main()
