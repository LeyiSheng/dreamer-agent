"""Fixed-checkpoint task-origin reward screening on untouched training-excluded sources.

This is a pre-RL diagnostic, not a final acceptance report. Never loads test data.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import cv2,numpy as np,torch
from .provenance import sha256
from .export_vpt import load_backend
from .learning import HistoryPolicy
from .reward import TemporalRewardCompletionHeads
from .prepare_reward_labels import clip_labels
from .calibration import select_threshold,metrics


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 if a.device.startswith('cuda'):torch.cuda.set_device(torch.device(a.device))
 out=a.output;out.mkdir(exist_ok=False,parents=True)
 source=Path('outputs/p1c_full_corpus_v1/seed_0/latest.pt')
 # Atomic producer replacement permits copying one immutable file descriptor.
 with source.open('rb') as src,(out/'reward_snapshot.pt').open('xb') as dst:shutil.copyfileobj(src,dst)
 bp=Path('outputs/p1b_task_bc_v1/seed_0/bc.pt')
 ck=torch.load(out/'reward_snapshot.pt',map_location='cpu',weights_only=True)
 assert ck['metadata']['bc_sha256']==sha256(bp)
 audit=Path('outputs/p1c_full_corpus_v1/audit/training_records.jsonl')
 assert sha256(audit)==ck['metadata']['audit_sha256']
 candidates={}
 for line in audit.open():
  r=json.loads(line)
  if r['split']!='calibration' or r['status']!='eligible_pending_full_decode':continue
  ev=[e for e in r['events'] if e.get('accepted') and e['task']=='obtain_log' and 8<=e['frame']<r['frames']-9]
  if not ev:continue
  key=r['source_session'].rsplit('-',1)[-1]
  candidates.setdefault(key,[]).append((r,ev[0]['frame']))
 ordered=sorted(candidates,key=lambda k:hashlib.sha256(('rl-screen-v1:'+k).encode()).hexdigest())[:32]
 records=[sorted(candidates[k],key=lambda x:x[0]['stem'])[0] for k in ordered]
 if len(records)<4:raise ValueError('Insufficient distinct calibration sessions')
 protocol=dict(snapshot_sha256=sha256(out/'reward_snapshot.pt'),bc_sha256=sha256(bp),
  updates=ck.get('updates'),horizon=16,seed=20260918,selection_sessions=ordered[::2],confirmation_sessions=ordered[1::2],
  sources=[r['stem'] for r,f in records],root='event_minus_8 with zero BC/reward state',
  warning='event-centered enriched screen plus source background; not representative final calibration',rl_ready=False)
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 spec,encode,action,extractor=load_backend(Path('outputs/acceptance_20260913/assets.json'),Path('outputs/acceptance_20260913/action_probe.json'),a.device)
 bc=torch.load(bp,map_location='cpu',weights_only=True)
 assert extractor==bc['metadata']['feature_spec']['extractor_sha256']
 policy=HistoryPolicy(**bc['metadata']['config']).to(a.device).eval().requires_grad_(False);policy.load_state_dict(bc['model'])
 head=TemporalRewardCompletionHeads(128,5).to(a.device).eval().requires_grad_(False);head.load_state_dict(ck['model'])
 scores=[]
 for index,(r,event) in enumerate(records):
  raw=Path(r['actions_path']);assert sha256(raw)==r['actions_sha256']
  rows=[json.loads(l) for l in raw.read_text().splitlines()]
  start=event-8;windows=[(start,start+17,'event')]
  accepted={e['frame'] for e in r['events'] if e.get('accepted') and e['task']=='obtain_log'}
  for c in r['clips']:
   if c['background'] and c['stop']-c['start']>=17 and not any(c['start']<=e<c['start']+17 for e in accepted) and abs(c['start']-start)>17:
    windows.append((c['start'],c['start']+17,'background'));break
  wanted={i for b,e,k in windows for i in range(b,e)};features={};vh=sha256(r['video_path'])
  cap=cv2.VideoCapture(r['video_path']);i=0
  try:
   while i<=max(wanted):
    ok,bgr=cap.read()
    if not ok:raise ValueError('Decode ended before selected clip')
    if i in wanted:
     features[i]=encode(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
     if not np.isfinite(features[i]).all():raise ValueError('Nonfinite tokenizer feature; stop before reward scoring')
    i+=1
  finally:cap.release()
  assert sha256(r['video_path'])==vh
  for b,e,kind in windows:
   f=torch.from_numpy(np.stack([features[i] for i in range(b,e)])).to(a.device)
   acts=np.stack([spec.to_local(np.asarray(action(row),dtype=np.int64)) for row in rows[b:e]])
   past=np.zeros((len(f),sum(spec.action_sizes)),np.float32);offset=0
   for k,n in enumerate(spec.action_sizes):past[1:,offset:offset+n]=np.eye(n,dtype=np.float32)[acts[:-1,k]];offset+=n
   labels=clip_labels(rows,r['events'],b,e,'obtain_log',0)
   with torch.no_grad():
    h=policy.encode_history(f[None],torch.from_numpy(past)[None].to(a.device))
    p=head(h[:,:-1],h[:,1:],torch.zeros(1,len(f)-1,dtype=torch.long,device=a.device))['completion'][0].sigmoid().cpu().tolist()
   for j,prob in enumerate(p):
    label=labels[j]['completion']
    if label is not None:scores.append(dict(score=prob,label=label,session=r['source_session'].rsplit('-',1)[-1],stem=r['stem'],frame=b+j,kind=kind))
  (out/'scores.json').write_text(json.dumps(scores))
  print('SOURCE',index+1,len(records),r['stem'],flush=True)
 select=[r for r in scores if r['session'] in ordered[::2]];confirm=[r for r in scores if r['session'] in ordered[1::2]]
 chosen=select_threshold([r['score'] for r in select],[r['label'] for r in select])
 result=dict(selection=chosen,selection_n=len(select),confirmation_n=len(confirm),
   selection_half=metrics([r['score'] for r in select],[r['label'] for r in select],.5),
   confirmation_half=metrics([r['score'] for r in confirm],[r['label'] for r in confirm],.5),rl_ready=False)
 if chosen['threshold'] is not None:result['confirmation']=metrics([r['score'] for r in confirm],[r['label'] for r in confirm],chosen['threshold'])
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
