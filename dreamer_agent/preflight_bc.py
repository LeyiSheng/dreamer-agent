"""Check a real P1a dataset and one backward pass without optimizer updates."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import OfflineWindows
from .learning import HistoryPolicy, bc_loss
from .provenance import sha256,write_report


def run(manifest,output,*,context=32,hidden=128,batch_size=8,seed=0,device="cpu"):
    train=OfflineWindows(manifest,"train",context,8)
    cal=OfflineWindows(manifest,"calibration",context,8)
    m=train.manifest
    if m.get("policy_mode")!="unconditioned":
        raise ValueError("This preflight requires task-free BC data")
    if any(r["split"]=="test" for r in m["episodes"]):
        raise ValueError("Launch-set preflight must not include test data")
    torch.manual_seed(seed)
    config=dict(feature_dim=m["feature_spec"]["dimension"],past_action_dim=sum(train.spec.action_sizes),
        tasks=1,action_sizes=train.spec.action_sizes,hidden=hidden,targets=8,task_conditioned=False)
    model=HistoryPolicy(**config).to(device)
    batch={k:v.to(device) for k,v in next(iter(DataLoader(train,batch_size=batch_size))).items()}
    logits=model(batch["features"],batch["past_actions"],batch["tasks"])
    loss=bc_loss(logits,batch["targets"],batch["valid"])
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite launch loss")
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise ValueError("Missing/nonfinite trainable gradient")
    if not any(g.abs().sum()>0 for g in grads):
        raise ValueError("All gradients zero")
    # Training-only unconditional frequency reference, evaluated on held-out actions.
    baseline=[];joint_nll=0.
    for k,n in enumerate(train.spec.action_sizes):
        values=np.concatenate([ep["actions"][:,k] for ep in train.episodes])
        frequencies=np.bincount(values,minlength=n)
        probability=(frequencies+1)/(len(values)+n)
        heldout=np.concatenate([ep["actions"][:,k] for ep in cal.episodes])
        nll=float(-np.log(probability[heldout]).mean());joint_nll+=nll
        baseline.append(dict(factor=k,train_counts=frequencies.tolist(),validation_current_nll=nll,
            validation_modal_accuracy=float((heldout==frequencies.argmax()).mean())))
    report=dict(ready_for_bc_launch=True,optimizer_steps=0,trained_checkpoint_created=False,
        task_success_measured=False,manifest_sha256=sha256(manifest),model_config=config,
        train_frames=sum(len(ep["features"]) for ep in train.episodes),
        calibration_frames=sum(len(ep["features"]) for ep in cal.episodes),
        train_windows=len(train),calibration_windows=len(cal),
        initial_batch_mtp_nll=float(loss.detach()),uniform_factor_nll=sum(math.log(n) for n in train.spec.action_sizes),
        training_frequency_baseline=baseline,baseline_validation_current_joint_nll=joint_nll,
        task_embedding_present=model.task is not None,all_gradients_finite=True,
        model_parameters=sum(p.numel() for p in model.parameters()),
        device=device,seed=seed,context=context,code_sha256={name:sha256(Path(__file__).with_name(name))
            for name in ("preflight_bc.py","learning.py","dataset.py","train_bc.py","actions.py")},
        scope="Launch-set engineering preflight; not convergence, full-data deduplication or task capability")
    write_report(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--context",type=int,default=32)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cpu")
    result=run(**vars(p.parse_args()))
    print(json.dumps({k:result[k] for k in ("ready_for_bc_launch","train_frames","calibration_frames","initial_batch_mtp_nll","optimizer_steps")}),flush=True)


if __name__=="__main__":
    main()
