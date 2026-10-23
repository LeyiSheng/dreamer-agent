"""Train event/completion heads from reviewed labels and a frozen BC checkpoint.

No real evaluation data, unreviewed labels, test-set tuning or imagination.
Calibration thresholds are diagnostic and do not unlock RL.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import OfflineWindows
from .learning import HistoryPolicy
from .reward import RewardCompletionHeads, supervised_head_loss, frozen_history
from .calibration import metrics, select_threshold
from .provenance import sha256, write_report


def load_labels(path, manifest):
    root=Path(path).resolve().parent
    m=json.loads(Path(path).read_text())
    if m.get("schema_version")!=1 or m.get("origin")!="offline" or m.get("reviewed") is not True:
        raise ValueError("Reviewed offline reward labels required")
    if m["feature_manifest_sha256"]!=sha256(manifest):
        raise ValueError("Labels refer to a different feature manifest")
    features=json.loads(Path(manifest).read_text())
    records={r["episode_id"]:r for r in features["episodes"]}
    labels={}
    for r in m["episodes"]:
        name=r["episode_id"]
        if name not in records or name in labels:
            raise ValueError("Unknown/duplicate label episode")
        p=root/r["path"]
        if sha256(p)!=r["sha256"]:
            raise ValueError("Label hash mismatch")
        rows=json.loads(p.read_text())
        with np.load(Path(manifest).resolve().parent/records[name]["path"],allow_pickle=False) as data:
            tasks=data["tasks"]
        if len(rows)!=len(tasks):
            raise ValueError("Labels must cover exactly the feature clip")
        for i,row in enumerate(rows):
            if set(row)!={"task","event","completion"} or type(row["task"]) is not int or row["task"]!=int(tasks[i]):
                raise ValueError("Task/label mismatch")
            if any(row[k] is not None and (type(row[k]) is not int or row[k] not in (0,1)) for k in ("event","completion")):
                raise ValueError("Invalid reward label")
        labels[name]=rows
    if set(labels)!=set(records):
        raise ValueError("Missing label episodes; use explicit null for unknown")
    return labels


def windows(data, annotations, policy, device):
    names=[r["episode_id"] for r in data.manifest["episodes"] if r["split"]==data.split]
    groups=[r["source_episode_id"] for r in data.manifest["episodes"] if r["split"]==data.split]
    for index,(episode,start) in enumerate(data.windows):
        batch={k:v[None].to(device) for k,v in data[index].items()}
        h=frozen_history(policy,batch["features"],batch["past_actions"])[0]
        tasks=batch["tasks"][0]
        length=min(data.context,len(data.episodes[episode]["features"])-start)
        if length<2:
            continue
        # Same reset/context convention as BC; never span a context/clip boundary.
        current,next_h=h[:length-1],h[1:length]
        valid_task=tasks[:length-1]==tasks[1:length]
        labels,masks={},{}
        rows=annotations[names[episode]][start:start+length-1]
        for name in ("event","completion"):
            labels[name]=torch.tensor([r[name] if r[name] is not None else 0 for r in rows],dtype=torch.float32,device=device)
            masks[name]=torch.tensor([r[name] is not None for r in rows],dtype=torch.bool,device=device)&valid_task
        yield current,next_h,tasks[:length-1],labels,masks,groups[episode]


def calibrate(records, tasks):
    reports=[]
    for task in range(tasks):
        result={"task":task,"heads":{}}
        for head in ("event","completion"):
            rows=[r for r in records if r["task"]==task and r["head"]==head]
            if not rows:
                result["heads"][head]=dict(threshold=None,passed=False,reason="No labeled calibration transitions")
                continue
            scores=[r["score"] for r in rows]; labels=[r["label"] for r in rows]
            selection=select_threshold(scores,labels)
            # A high-precision all-positive set cannot establish false-positive behavior.
            if len(set(labels))<2:
                selection=dict(threshold=None,passed=False,reason="Both positive and negative calibration examples required")
            fixed=metrics(scores,labels,.5)
            tail=np.argsort(scores)[-min(100,len(scores)):]
            per_session={}
            grouped={}
            for row in rows:
                grouped.setdefault(row["session"],[]).append(row)
            for session,subset in sorted(grouped.items()):
                per_session[session]=metrics([r["score"] for r in subset],[r["label"] for r in subset],selection["threshold"] if selection["threshold"] is not None else .5)
            result["heads"][head]=dict(selection=selection,at_half=fixed,
                top100_errors=sum(labels[i]==0 for i in tail),top100_count=len(tail),
                per_session=per_session,independent_confirmation_passed=False)
        reports.append(result)
    return dict(tasks=reports,rl_ready=False,scope="Calibration selection only; correlated frames; per-session diagnostics, no independent confirmation")



def calibration_with_confirmation(records, tasks, sessions):
    """Predefined session partition; confirmation never participates in selection."""
    sessions=sorted(set(sessions))
    # This partition depends only on session IDs, never scores or labels.
    confirmation=set(sessions[1::2])
    selection=set(sessions[::2])
    selected=[r for r in records if r["session"] in selection]
    heldout=[r for r in records if r["session"] in confirmation]
    result=calibrate(selected,tasks)
    result["partition"]=dict(selection_sessions=sorted(selection),confirmation_sessions=sorted(confirmation),
        rule="sorted calibration session IDs; even positions select, odd positions confirm")
    for task_result in result["tasks"]:
        task=task_result["task"]
        for head,report in task_result["heads"].items():
            threshold=report.get("selection",{}).get("threshold")
            rows=[r for r in heldout if r["task"]==task and r["head"]==head]
            check=dict(passed=False,reason="No selected threshold or independent labeled confirmation set")
            if threshold is not None and rows:
                labels=[r["label"] for r in rows]
                measured=metrics([r["score"] for r in rows],labels,threshold)
                interval=measured["precision_interval"]
                enough=measured["tp"]+measured["fp"]>=30
                check=dict(metrics=measured,passed=bool(len(set(labels))==2 and enough and interval and interval[0]>=.95),
                    scope="Fixed-threshold held-out session check; frame-IID interval is diagnostic")
            report["confirmation"]=check
            report["independent_confirmation_passed"]=check["passed"]
    result["scope"]="Session-separated selection and fixed-threshold confirmation; temporal correlation not resolved; no RL authorization"
    return result

def run(manifest,labels,bc,output,*,epochs=1,hidden=128,lr=3e-4,seed=0,device="cpu"):
    if epochs<1 or hidden<1 or not np.isfinite(lr) or lr<=0:
        raise ValueError("Invalid reward training configuration")
    torch.manual_seed(seed)
    annotations=load_labels(labels,manifest)
    checkpoint=torch.load(bc,map_location="cpu",weights_only=True)
    metadata=checkpoint["metadata"]
    if metadata["manifest_sha256"]!=sha256(manifest) or metadata["status"]!="supervised_only":
        raise ValueError("BC checkpoint provenance mismatch")
    policy=HistoryPolicy(**metadata["config"])
    policy.load_state_dict(checkpoint["model"],strict=True)
    policy=policy.to(device).eval().requires_grad_(False)
    context=metadata["context"]
    if context<2:
        raise ValueError("Reward transitions require context >=2")
    train=OfflineWindows(manifest,"train",context,metadata["config"]["targets"])
    calibration=OfflineWindows(manifest,"calibration",context,metadata["config"]["targets"])
    if metadata["action_spec_sha256"]!=train.spec.fingerprint or metadata["feature_spec"]!=train.manifest["feature_spec"]:
        raise ValueError("BC feature/action contract mismatch")
    head_config=dict(history_dim=metadata["config"]["hidden"],tasks=train.manifest["tasks"],hidden=hidden)
    head=RewardCompletionHeads(**head_config).to(device)
    # Reject unsupported tasks before taking any optimizer step.
    coverage={task:{k:[0,0] for k in ("event","completion")} for task in range(head_config["tasks"])}
    # Frozen readout is identical across epochs; retain this run only, never reuse old caches.
    train_windows=list(windows(train,annotations,policy,device))
    for _,_,task,y,mask,_ in train_windows:
        for k in mask:
            for t in range(head_config["tasks"]):
                vals=y[k][mask[k]&(task==t)].long()
                for value in (0,1):
                    coverage[t][k][value]+=int((vals==value).sum())
    if any(min(counts)==0 for heads in coverage.values() for counts in heads.values()):
        raise ValueError("Every task/head needs observed positive and negative training transitions: "+str(coverage))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    config=dict(schema_version=1,head_config=head_config,manifest_sha256=sha256(manifest),
        labels_sha256=sha256(labels),bc_sha256=sha256(bc),bc_context=context,
        epochs=epochs,lr=lr,seed=seed,device=device,coverage=coverage,
        code_sha256={name:sha256(Path(__file__).with_name(name)) for name in ("reward.py","learning.py","dataset.py","train_reward.py","calibration.py")},
        plan_sha256=sha256(Path(__file__).resolve().parents[1]/"docs/PLAN_V6.md"),
        status="supervised_reward_only",rl_ready=False,history_cache="in-memory this run only; frozen BC readout",weight_decay=0.01)
    write_report(out/"config.json",config)
    opt=torch.optim.AdamW(head.parameters(),lr=lr)
    for epoch in range(epochs):
        head.train();total,count=0.,0
        for h,nxt,task,y,mask,_ in train_windows:
            if not any(m.any() for m in mask.values()):
                continue
            loss=supervised_head_loss(head(h,nxt,task),y,mask)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reward loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=loss.item();count+=1
        with (out/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(epoch=epoch+1,mean_window_bce=total/count,windows=count,task_success_measured=False))+"\n")
    if any(not torch.equal(v.detach().cpu(),checkpoint["model"][k]) for k,v in policy.state_dict().items()):
        raise RuntimeError("Frozen BC parameters changed")
    if any(not torch.isfinite(v).all() for v in head.state_dict().values()):
        raise RuntimeError("Nonfinite reward parameters")
    head.eval();scores=[]
    with torch.no_grad():
        for h,nxt,task,y,mask,group in windows(calibration,annotations,policy,device):
            logits=head(h,nxt,task)
            for name in logits:
                for i in torch.where(mask[name])[0].tolist():
                    scores.append(dict(task=int(task[i]),head=name,score=float(logits[name][i].sigmoid()),label=int(y[name][i]),session=group))
    write_report(out/"calibration_scores.json",scores)
    write_report(out/"calibration.json",calibration_with_confirmation(scores,head_config["tasks"],
        [r["source_episode_id"] for r in calibration.manifest["episodes"] if r["split"]=="calibration"]))
    torch.save(dict(model=head.state_dict(),metadata=config),out/"reward.pt")
    return out/"reward.pt"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","labels","bc","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cpu")
    run(**vars(p.parse_args()))


if __name__=="__main__":
    main()
