"""Real frozen-model 4-step integration test, zero optimizer updates."""
import importlib.util,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from .provenance import sha256
from .actions import ActionSpec
from .contracts import Episode
from .learning import HistoryPolicy
from .reward import TemporalRewardCompletionHeads
from .world_model import FrozenWorldAdapter
from .imagination_backend import OfflineRoot,FrozenFeatureBackend
from .vpt_rl import load_vpt,VPTReadout,VPTToWorldAction,VPTFeatureBackend
from .rl_pixels import generated_to_p1_pixels,P1PixelEncoder
from .rl_roots import tensor_hash


def build_backend(out):
 torch.set_num_threads(1);torch.cuda.set_device(0)
 out=Path(out);out.mkdir(exist_ok=False,parents=True)
 manifest_path=Path('outputs/p2_vpt_inputs_20260918/bc_manifest.json');m=json.loads(manifest_path.read_text())
 video,raw=m['sources'][0],m['sources'][1]
 assert sha256(video['path'])==video['sha256'] and sha256(raw['path'])==raw['sha256']
 assets=json.loads(Path('outputs/acceptance_20260913/assets.json').read_text())
 for entry in assets['files']:
  if sha256(entry['path'])!=entry['sha256']:raise ValueError('WM assets changed')
 snapshot=Path(assets['files'][0]['path']).parents[1]
 sys.path.insert(0,str(Path('Dreamerv4-MC').resolve()))
 from src.modules.tokenizer import CausalTokenizer
 from src.modules.dynamic_model import DynamicModel
 from src.modules.actokenizer import MineCraftActionTokenizer
 from src.inference.infer import generate_one_frame
 models=[]
 for cls,sub in [(CausalTokenizer,'tokenizer'),(DynamicModel,'dynamic')]:
  net,info=cls.from_pretrained(str(snapshot),subfolder=sub,local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
  if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Strict WM loading failed')
  models.append(net.cuda().eval().requires_grad_(False))
 tokenizer,dynamics=models
 spec=ActionSpec.from_dict(json.loads(Path('outputs/acceptance_20260913/action_probe.json').read_text())['action_spec'])
 world=FrozenWorldAdapter(tokenizer,dynamics,generate_one_frame,spec,seed=0)
 encoder=P1PixelEncoder(tokenizer);codec=MineCraftActionTokenizer()
 cap=cv2.VideoCapture(video['path']);ok,bgr=cap.read();cap.release()
 if not ok:raise ValueError('Cannot decode root')
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 pixels=torch.nn.functional.pad(torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/127.5-1,(0,0,12,12))[None,None]
 row=json.loads(Path(raw['path']).read_text().splitlines()[0])
 tokens=torch.tensor(codec.get_action_index_from_actiondict(codec.json_action_to_env_action(row,hotbar=True)[0],include_gui=True),device='cuda',dtype=torch.long)[None]
 f=encoder(pixels)[None]
 past=torch.zeros(1,sum(spec.action_sizes),device='cuda')
 ep=Episode(**m['episodes'][0])
 # This diagnostic explicitly labels physical alignment as unverified.
 alignment=out/'alignment_contract.json';alignment.write_text(json.dumps(dict(status='diagnostic_only',physical_semantics_verified=False,convention='same-index context conditioning as existing adapter; requires semantic validation')))
 root_manifest=dict(schema_version=1,episode=vars(ep),frame=0,task_start_frame=0,feature_contract_sha256=sha256('dreamer_agent/rl_pixels.py'),alignment_evidence_sha256=sha256(alignment),
   sources=[video,raw],tensors={k:tensor_hash(v) for k,v in dict(pixels=pixels,frame_actions=tokens,features=f,past_actions=past).items()})
 rp=out/'root.json';rp.write_text(json.dumps(root_manifest,indent=2))
 root=OfflineRoot(ep,pixels,tokens,f,dict(manifest_path=str(rp.resolve()),manifest_sha256=sha256(rp)),past)
 bp=Path('outputs/p1b_task_bc_v1/seed_0/bc.pt');bc=torch.load(bp,map_location='cpu',weights_only=True)
 policy=HistoryPolicy(**bc['metadata']['config']).cuda().eval().requires_grad_(False);policy.load_state_dict(bc['model'])
 checkpoint=Path('outputs/p2_reward_screen_20260918_v2/reward_snapshot.pt');ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert ck['metadata']['bc_sha256']==sha256(bp)
 reward=TemporalRewardCompletionHeads(128,5).cuda().eval().requires_grad_(False);reward.load_state_dict(ck['model'])
 rb=FrozenFeatureBackend(world,policy,reward,encoder,generated_to_p1_pixels,consistency=lambda *args:False)
 # Consistency intentionally unapproved: no model reward is accepted here.
 vpt,mapper,xf=load_vpt(Path('checkpoints/foundation-model-3x.model'),Path('checkpoints/bc-early-game-3x.weights'),
  model_sha256=m['model_sha256'],weights_sha256=m['weights_sha256'],device='cuda:0')
 backend=VPTFeatureBackend(rb,vpt,VPTToWorldAction(mapper,xf,codec,spec,'cuda:0'))
 return backend,vpt,root,checkpoint,m

def main():
 out=Path('outputs/p2_vpt_integration_20260918')
 backend,vpt,root,checkpoint,m=build_backend(out)
 actor=VPTReadout(vpt).cuda().requires_grad_(True)
 history=backend.reset(root,0,0);rows=[]
 for i in range(4):
  with torch.no_grad():act=torch.stack([x.argmax(-1) for x in actor(history,torch.tensor(0,device='cuda'))])
  result=backend.step(act)
  rows.append(dict(step=i,actions=act.tolist(),completion_score=result.completion_probability,finite_history=bool(torch.isfinite(result.history).all())))
  history=result.history
  print('STEP',i,rows[-1],flush=True)
 logits=actor(history,torch.tensor(0,device='cuda'));loss=-sum(x.log_softmax(-1)[0] for x in logits);loss.backward()
 report=dict(kind='real_models_integration_diagnostic',optimizer_steps=0,world_steps=4,rows=rows,
  original_vpt_sha256=m['weights_sha256'],reward_sha256=sha256(checkpoint),
  actor_gradients_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in actor.parameters()),
  frozen_gradients_absent=all(p.grad is None for net in backend.frozen_modules for p in net.parameters()),
  reward_consistency_accepted=False,physical_action_semantics_verified=False,rl_ready=False)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
