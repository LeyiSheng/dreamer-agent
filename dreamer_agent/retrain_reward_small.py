"""Small, paired reward retraining experiment; full corpus is conditionally gated."""
import json, argparse, copy
from pathlib import Path
import numpy as np
import torch
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.reward import RewardCompletionHeads, TemporalRewardCompletionHeads
from dreamer_agent.provenance import sha256
from dreamer_agent.contracts import Episode,validate_dataset

HEADS=("event","completion")
def stats(logits,labels):
 out={}
 for task in range(5):
  out[task]={}
  for head in HEADS:
   z=np.array(logits[task][head]);y=np.array(labels[task][head])
   if not len(y):out[task][head]={"n":0,"ap":None,"positives":0};continue
   order=np.argsort(-z,kind="stable");zs=z[order];ys=y[order];ends=np.r_[np.flatnonzero(np.diff(zs)!=0),len(z)-1]
   tp=np.cumsum(ys)[ends];n=ends+1;rec=tp/max(y.sum(),1)
   ap=float(np.sum(np.diff(np.r_[0,rec])*tp/n)) if y.sum() else None
   p=1/(1+np.exp(-np.clip(z,-60,60)))
   out[task][head]=dict(n=len(y),positives=int(y.sum()),ap=ap,bce=float(np.mean(np.logaddexp(0,z)-y*z)),brier=float(np.mean((p-y)**2)),
    recall_half=float(((p>=.5)&(y==1)).sum()/max(y.sum(),1)),fp_half=int(((p>=.5)&(y==0)).sum()))
 return out
def batch(items,ids):
 rows=[items[int(i)] for i in ids];length=max(len(x["h"]) for x in rows)
 h=torch.zeros(len(rows),length,128,device="cuda");n=torch.zeros_like(h);t=torch.zeros(len(rows),length,dtype=torch.long,device="cuda")
 y={k:torch.zeros(len(rows),length,device="cuda") for k in HEADS};mask={k:torch.zeros(len(rows),length,dtype=torch.bool,device="cuda") for k in HEADS}
 for i,r in enumerate(rows):
  l=len(r["h"]);h[i,:l]=r["h"];n[i,:l]=r["n"];t[i,:l]=r["task"]
  for k in HEADS:y[k][i,:l]=r["y"][k];mask[k][i,:l]=r["mask"][k]
 return h,n,t,y,mask
@torch.no_grad()
def evaluate(model,items):
 model.eval();scores={t:{k:[] for k in HEADS} for t in range(5)};labels=copy.deepcopy(scores)
 for start in range(0,len(items),16):
  ids=np.arange(start,min(start+16,len(items)));h,n,t,y,m=batch(items,ids);z=model(h,n,t)
  for task in range(5):
   for k in HEADS:
    keep=m[k]&(t==task);scores[task][k].extend(z[k][keep].cpu().tolist());labels[task][k].extend(y[k][keep].cpu().tolist())
 return stats(scores,labels),dict(logits=scores,labels=labels)
def load_items(records,annotations,mp,policy,sizes):
 out=[]
 for r in records:
  p=mp.parent/r["path"];assert sha256(p)==r["sha256"]
  with np.load(p,allow_pickle=False) as z:f=z["features"].astype(np.float32);a=z["actions"];tasks=z["tasks"];frames=z["frames"]
  assert set(z.files)=={"features","actions","tasks","frames"} and np.isfinite(f).all()
  assert (np.diff(frames)==1).all()
  pp=np.zeros((len(a),sum(sizes)),np.float32);offset=0
  for k,num in enumerate(sizes):
   pp[1:,offset:offset+num]=np.eye(num,dtype=np.float32)[a[:-1,k]];offset+=num
  hidden=[]
  with torch.no_grad():
   for start in range(0,len(f),32):
    hidden.append(policy.encode_history(torch.from_numpy(f[start:start+32])[None].cuda(),torch.from_numpy(pp[start:start+32])[None].cuda())[0])
  hs=torch.cat(hidden);ls=annotations[r["episode_id"]];assert len(ls)==len(f)
  item=dict(h=hs[:-1],n=hs[1:],task=int(tasks[0]),session=r["source_episode_id"],episode=r["episode_id"],y={},mask={})
  assert (tasks==tasks[0]).all()
  for k in HEADS:
   item["y"][k]=torch.tensor([x[k] if x[k] is not None else 0 for x in ls[:-1]],device="cuda",dtype=torch.float32)
   item["mask"][k]=torch.tensor([x[k] is not None and i%32!=31 for i,x in enumerate(ls[:-1])],device="cuda")
  out.append(item)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);a=ap.parse_args()
 torch.set_num_threads(1);seed=a.seed;torch.manual_seed(seed)
 mp=Path("outputs/p1b_task_features_20260914_v2/manifest.json");lp=Path("outputs/p1c_labels_20260915/reward_labels.json")
 m=json.loads(mp.read_text());lm=json.loads(lp.read_text());assert lm["feature_manifest_sha256"]==sha256(mp)
 validate_dataset([Episode(r["episode_id"],r["source_episode_id"],r["split"],r["origin"]) for r in m["episodes"]])
 records=[]
 for task in m["task_names"]:
  rows=[r for r in m["episodes"] if r["split"]=="train" and r["event_evidence"]["task"]==task]
  records+=sorted(rows,key=lambda r:__import__("hashlib").sha256(("20260915:"+r["episode_id"]).encode()).hexdigest())[:64]
 records=sorted(records,key=lambda r:r["episode_id"])
 cal=[r for r in m["episodes"] if r["split"]=="calibration"];allr=records+cal;needed={r["episode_id"] for r in allr};ann={}
 for r in lm["episodes"]:
  if r["episode_id"] in needed:
   p=lp.parent/r["path"];assert sha256(p)==r["sha256"];ann[r["episode_id"]]=json.loads(p.read_text())
 bp=Path(f"outputs/p1b_task_bc_v1/seed_{seed}/bc.pt");ck=torch.load(bp,map_location="cpu",weights_only=True)
 assert ck["metadata"]["manifest_sha256"]==sha256(mp) and ck["metadata"]["feature_spec"]==m["feature_spec"]
 policy=HistoryPolicy(**ck["metadata"]["config"]).cuda().eval().requires_grad_(False);policy.load_state_dict(ck["model"])
 data=load_items(allr,ann,mp,policy,ck["metadata"]["config"]["action_sizes"])
 train=data[:len(records)];caldata=data[len(records):];sessions=sorted({x["session"] for x in caldata});select=[x for x in caldata if x["session"] in sessions[::2]];confirm=[x for x in caldata if x["session"] in sessions[1::2]]
 out=Path(f"outputs/p1c_retrain_small_v2/seed_{seed}");out.mkdir(exist_ok=False)
 provenance=dict(bc_sha256=sha256(bp),manifest_sha256=sha256(mp),labels_sha256=sha256(lp),script_sha256=sha256(__file__),plan_sha256=sha256("docs/PLAN_V6.md"),train_episode_ids=[x["episode"] for x in train],selection_sessions=sessions[::2],confirmation_sessions=sessions[1::2])
 (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
 bytask=[np.array([i for i,x in enumerate(train) if x["task"]==t]) for t in range(5)]
 weights={k:torch.ones(5,device="cuda") for k in HEADS}
 for t in range(5):
  for k in HEADS:
   ys=torch.cat([train[i]["y"][k][train[i]["mask"][k]] for i in bytask[t]])
   weights[k][t]=min(10.,float(torch.sqrt((len(ys)-ys.sum())/ys.sum().clamp_min(1))))
 report={}
 torch.manual_seed(seed);initial=RewardCompletionHeads(128,5,128).cuda()
 for arm in ("control","fixed"):
  torch.manual_seed(seed)
  model=copy.deepcopy(initial) if arm=="control" else TemporalRewardCompletionHeads(128,5,128).cuda()
  if arm=="fixed":
   state=model.state_dict()
   for k,v in initial.state_dict().items():state[k]=v.detach().clone()
   model.load_state_dict(state)
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);rng=np.random.default_rng(seed+20260915);best=-1;best_state=None;history=[]
  for epoch in range(8):
   model.train();losses=[]
   for step in range(40):
    if arm=="control":ids=(np.arange(16)+(epoch*40+step)*16)%len(train)
    else:ids=np.array([rng.choice(bytask[int(t)]) for t in rng.integers(0,5,16)])
    h,n,t,y,mask=batch(train,ids);z=model(h,n,t);loss=0
    for k in HEADS:
     v=torch.nn.functional.binary_cross_entropy_with_logits(z[k],y[k],reduction="none")
     if arm=="fixed":v=v*torch.where(y[k]>0,weights[k][t],1.)
     loss=loss+v[mask[k]].mean()/2
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss))
   metrics,_=evaluate(model,select)
   eligible=[x["ap"] for hs in metrics.values() for x in hs.values() if x["positives"]>=5 and x["ap"] is not None]
   criterion=float(np.mean(eligible));row=dict(epoch=epoch+1,loss=float(np.mean(losses)),selection=metrics,selection_macro_ap=criterion)
   history.append(row)
   with (out/f"{arm}_metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
   if criterion>best:best=criterion;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   print(seed,arm,epoch+1,round(criterion,4),flush=True)
  torch.save(dict(model=model.state_dict(),provenance=provenance,arm=arm),out/f"{arm}_last.pt")
  model.load_state_dict(best_state);model.eval();metrics,scores=evaluate(model,confirm)
  torch.save(dict(model=best_state,provenance=provenance,arm=arm,epoch=best_epoch),out/f"{arm}_best.pt")
  report[arm]=dict(best_epoch=best_epoch,selection_macro_ap=best,confirmation=metrics,positive_weights={k:v.tolist() for k,v in weights.items()} if arm=="fixed" else None)
  if arm=="fixed":
   permuted=[];prng=np.random.default_rng(915)
   for item in confirm:
    perm=torch.as_tensor(prng.permutation(len(item["h"])),device="cuda")
    permuted.append(dict(item,h=item["h"][perm],n=item["n"][perm]))
   pm,_=evaluate(model,permuted)
   report[arm]["confirmation_time_permuted"]=pm
  (out/f"{arm}_confirmation_scores.json").write_text(json.dumps(scores))
  (out/"report.json").write_text(json.dumps(report,indent=2))
 assert sha256(bp)==provenance["bc_sha256"]
 print("DONE",seed,flush=True)
if __name__=="__main__":main()
