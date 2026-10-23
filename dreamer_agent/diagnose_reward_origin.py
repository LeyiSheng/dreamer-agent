"""Read-only reward origin ablation on fixed previously inspected calibration data."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from .provenance import sha256
from .export_vpt import load_backend
from .learning import HistoryPolicy
from .reward import TemporalRewardCompletionHeads
from .prepare_reward_labels import clip_labels

def main():
 torch.set_num_threads(1);torch.cuda.set_device(0);out=Path('outputs/p2_reward_diagnosis_20260921');out.mkdir(exist_ok=False)
 old=Path('outputs/p2_reward_screen_20260918_v2');protocol=json.loads((old/'protocol.json').read_text());wanted=set(protocol['sources'][::2][:8]);records=[]
 for l in Path('outputs/p1c_full_corpus_v1/audit/training_records.jsonl').open():
  r=json.loads(l)
  if r['stem'] in wanted:records.append(r)
 assert len(records)==8 and all(r['split']=='calibration' for r in records)
 bp=Path('outputs/p1b_task_bc_v1/seed_0/bc.pt');bc=torch.load(bp,map_location='cpu',weights_only=True)
 spec,encode,action,extractor=load_backend(Path('outputs/acceptance_20260913/assets.json'),Path('outputs/acceptance_20260913/action_probe.json'),'cuda:0')
 assert extractor==bc['metadata']['feature_spec']['extractor_sha256']
 policy=HistoryPolicy(**bc['metadata']['config']).cuda().eval().requires_grad_(False);policy.load_state_dict(bc['model'])
 heads={};bindings={}
 for name,path in [('full_snapshot',old/'reward_snapshot.pt'),('small_best',Path('outputs/p1c_retrain_small_v2/seed_0/fixed_best.pt'))]:
  ck=torch.load(path,map_location='cpu',weights_only=True);md=ck.get('metadata',ck.get('provenance'));assert md['bc_sha256']==sha256(bp)
  net=TemporalRewardCompletionHeads(128,5).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model']);heads[name]=net;bindings[name]=dict(path=str(path),sha256=sha256(path))
 (out/'protocol.json').write_text(json.dumps(dict(sources=sorted(wanted),bindings=bindings,bc_sha256=sha256(bp),purpose='diagnostic only; previously inspected selection sessions; no threshold approval or training',modes=['reset8','earlier_clip_context']),indent=2))
 result=[]
 for r in records:
  assert sha256(r['actions_path'])==r['actions_sha256'];rows=[json.loads(l) for l in Path(r['actions_path']).read_text().splitlines()]
  event=next(e['frame'] for e in r['events'] if e.get('accepted') and e['task']=='obtain_log' and 8<=e['frame']<r['frames']-9)
  root=event-8;stop=event+9
  clips=[c for c in r['clips'] if c['task']=='obtain_log' and not c['background'] and c['start']<=event<c['stop']]
  start=min(root,clips[0]['start']) if clips else max(0,event-64)
  vh=sha256(r['video_path']);cap=cv2.VideoCapture(r['video_path']);features=[]
  try:
   for i in range(stop):
    ok,bgr=cap.read()
    if not ok:raise ValueError('Decode failed')
    if i>=start:features.append(encode(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
  finally:cap.release()
  assert sha256(r['video_path'])==vh
  f=np.stack(features);assert np.isfinite(f).all()
  acts=np.stack([spec.to_local(np.asarray(action(row),dtype=np.int64)) for row in rows[start:stop]])
  labels=clip_labels(rows,r['events'],root,stop,'obtain_log',0)
  for mode,b in [('reset8',root),('earlier_clip_context',start)]:
   offset=b-start;ff=torch.from_numpy(f[offset:]).cuda();aa=acts[offset:];past=np.zeros((len(ff),sum(spec.action_sizes)),np.float32);off=0
   for k,n in enumerate(spec.action_sizes):past[1:,off:off+n]=np.eye(n,dtype=np.float32)[aa[:-1,k]];off+=n
   with torch.no_grad():
    h=torch.cat([policy.encode_history(ff[j:j+32][None],torch.from_numpy(past[j:j+32])[None].cuda())[0] for j in range(0,len(ff),32)])[None]
    for name,net in heads.items():
     scores=net(h[:,:-1],h[:,1:],torch.zeros(1,len(ff)-1,dtype=torch.long,device='cuda'))['completion'][0].sigmoid().cpu().tolist()
     for frame in range(root,stop-1):
      label=labels[frame-root]['completion']
      if label is not None:result.append(dict(stem=r['stem'],mode=mode,head=name,frame=frame,label=label,score=scores[frame-b],context_start=b,original_clip_found=bool(clips),bc_boundary=(frame-b)%32==31))
  (out/'scores.json').write_text(json.dumps(result));print('SOURCE',r['stem'],flush=True)
 summary={}
 for name in heads:
  for mode in ['reset8','earlier_clip_context']:
   rr=[r for r in result if r['head']==name and r['mode']==mode and not r['bc_boundary']]
   pos=[r['score'] for r in rr if r['label']];neg=[r['score'] for r in rr if not r['label']]
   summary[name+'/'+mode]=dict(positive_n=len(pos),negative_n=len(neg),positive_median=float(np.median(pos)),negative_median=float(np.median(neg)),maximum=max(pos+neg),auroc=float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg])))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
