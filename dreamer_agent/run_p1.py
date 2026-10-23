"""Sequential P1 implementation: reviewed export -> BC -> frozen reward fit.

This runner does not start imagination or real-environment evaluation.
"""
import argparse
import json
from pathlib import Path
from .export_vpt import validate_annotations, load_backend, export
from .train_bc import run as train_bc
from .train_reward import run as train_reward
from .provenance import sha256, write_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--assets",type=Path,default=Path("outputs/acceptance_20260913/assets.json"))
    p.add_argument("--action-probe",type=Path,default=Path("outputs/acceptance_20260913/action_probe.json"))
    p.add_argument("--device",default="cuda")
    p.add_argument("--context",type=int,default=32)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--bc-epochs",type=int,default=1)
    p.add_argument("--reward-epochs",type=int,default=1)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--stage",choices=("bc","all"),default="bc")
    a=p.parse_args()
    source = validate_annotations(a.annotations)
    if a.stage == "all" and source.get("policy_mode") == "unconditioned":
        raise ValueError("Unconditioned BC has no reward labels; use --stage bc")
    if a.context<2 or min(a.hidden,a.batch_size,a.bc_epochs,a.reward_epochs)<1:
        raise ValueError("Invalid training configuration")
    a.output.mkdir(parents=True,exist_ok=False)
    status=dict(stage="export",completed=False,p1_passed=False,
                annotations_sha256=sha256(a.annotations))
    try:
        spec,encode,action,fingerprint=load_backend(a.assets,a.action_probe,a.device)
        manifest=export(a.annotations,a.output/"features",spec,encode,action,fingerprint)
        # Release frozen visual backend before allocating the BC trainer.
        del encode
        import gc,torch
        gc.collect()
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
        status["stage"]="bc"
        bc=train_bc(manifest,a.output/"bc",epochs=a.bc_epochs,context=a.context,
            hidden=a.hidden,batch_size=a.batch_size,seed=a.seed,device=a.device)
        status.update(stage="completed_bc",completed=True,bc_sha256=sha256(bc),
                      remaining="Task-conditioned BC, reward learning and real evaluation are separate stages")
        if a.stage == "all":
            status["stage"]="reward"
            status["completed"]=False
            reward=train_reward(manifest,manifest.parent/"reward_labels.json",bc,
                a.output/"reward",epochs=a.reward_epochs,hidden=a.hidden,seed=a.seed,device=a.device)
            status.update(stage="completed_supervised_pipeline",completed=True,
                reward_sha256=sha256(reward),
                remaining="World-model semantic acceptance and preregistered real BC evaluation; inspect reward confirmation before RL")
    except Exception as exc:
        status.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_report(a.output/"pipeline_report.json",status)


if __name__=="__main__":
    main()
