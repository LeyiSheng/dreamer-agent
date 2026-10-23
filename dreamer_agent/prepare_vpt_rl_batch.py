"""Small traceable VPT BC batch and actual-pixel forward preflight; no RL updates."""
import importlib.util,json
from pathlib import Path
import cv2,numpy as np,torch
from .provenance import sha256
from .vpt_rl import load_vpt,VPTReadout,VPTToWorldAction
from .actions import ActionSpec
from .rl_data import load_bc_batch


def main():
 torch.set_num_threads(1)
 out=Path('outputs/p2_vpt_inputs_20260918');out.mkdir(exist_ok=False)
 audit=Path('outputs/p1c_full_corpus_v1/audit/training_records.jsonl')
 for line in audit.open():
  r=json.loads(line)
  if r['split']=='train' and r['status']=='eligible_pending_full_decode' and r['frames']>=16:break
 else:raise ValueError('No audited train input')
 model=Path('checkpoints/foundation-model-3x.model');weights=Path('checkpoints/bc-early-game-3x.weights')
 binding=dict(model_sha256=sha256(model),weights_sha256=sha256(weights))
 vpt,mapper,xf=load_vpt(model,weights,**binding)
 codec_path=Path('Dreamerv4-MC/src/modules/actokenizer.py')
 spec=ActionSpec.from_dict(json.loads(Path('outputs/acceptance_20260913/action_probe.json').read_text())['action_spec'])
 assert sha256(codec_path)==spec.codec_sha256
 modspec=importlib.util.spec_from_file_location('verified_codec',codec_path);module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
 codec=module.MineCraftActionTokenizer()
 convert=VPTToWorldAction(mapper,xf,codec,spec,'cpu')
 # Exercise all button categories at neutral camera, all camera categories at neutral buttons.
 failed=[]
 for b,c in [(b,60) for b in range(8641)]+[(0,c) for c in range(121)]:
  try:convert(torch.tensor([b,c]))
  except Exception as exc:failed.append(dict(button=b,camera=c,error=str(exc)))
 rows=[json.loads(l) for l in Path(r['actions_path']).read_text().splitlines()]
 assert sha256(r['actions_path'])==r['actions_sha256']
 cap=cv2.VideoCapture(r['video_path']);features=[];targets=[];state=vpt.initial_state(1);maxdiff=0.
 try:
  for i in range(16):
   ok,bgr=cap.read()
   if not ok:raise ValueError('Short train clip')
   img=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_LINEAR)
   with torch.no_grad():
    (h,_),state=vpt.net({'img':torch.from_numpy(img)[None,None]},state,context={'first':torch.tensor([[i==0]])})
    features.append(h[0,0].clone())
   env,_=codec.json_action_to_env_action(rows[i],hotbar=True)
   factored=xf.env2policy(env)
   if factored['camera'].ndim==1:factored={k:v[None] for k,v in factored.items()}
   action=mapper.from_factored(factored)
   targets.append([int(np.asarray(action[k]).reshape(-1)[0]) for k in ('buttons','camera')])
 finally:cap.release()
 history=torch.stack(features)[None]
 tasks=torch.zeros(1,16,dtype=torch.long)
 head=VPTReadout(vpt).requires_grad_(True)
 scores=head(history,tasks)
 with torch.no_grad():original=vpt.pi_head(history)
 maxdiff=max(float((s-original[k].squeeze(-2)).abs().max()) for s,k in zip(scores,head.keys))
 loss=sum(-s.log_softmax(-1).gather(-1,torch.tensor(targets)[None,:,j,None]).mean() for j,s in enumerate(scores))
 loss.backward()
 torch.save(dict(history=history,tasks=tasks,targets=torch.tensor(targets)[None,:,None,:],mask=torch.ones(1,16,1,dtype=torch.bool)),out/'bc_batch.pt')
 sources=[dict(path=str(Path(p).resolve()),sha256=sha256(p)) for p in [r['video_path'],r['actions_path'],weights,model,codec_path]]
 manifest=dict(schema_version=1,feature_contract_sha256=sha256(__file__),episodes=[dict(episode_id=r['stem']+':0:16',source_episode_id=r['source_session'],split='train',origin='offline')],sources=sources,
 tensors=dict(path='bc_batch.pt',sha256=sha256(out/'bc_batch.pt')),frames=list(range(16)),preprocessing='RGB uint8 cv2 linear resize128; first true at frame0; sequential VPT recurrent forward',**binding)
 (out/'bc_manifest.json').write_text(json.dumps(manifest,indent=2))
 load_bc_batch(out/'bc_manifest.json',sha256(out/'bc_manifest.json'))
 report=dict(real_video=True,frames=16,source=r['stem'],strict_load=True,logit_max_difference=maxdiff,
   head_gradients_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in head.parameters()),
   backbone_gradients_absent=all(p.grad is None for p in vpt.parameters()),optimizer_steps=0,
   action_conversion_cases=8762,action_conversion_failures=failed,physical_action_semantics_verified=False,
   training_data_manifest_sha256=sha256(out/'bc_manifest.json'))
 (out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
