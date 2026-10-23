"""Fixed-world real-environment comparison. Pixel-only policy, isolated inventory scorer."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from .provenance import sha256

LOG_ITEMS={'log','log2','oak_log','birch_log','spruce_log','jungle_log','acacia_log','dark_oak_log'}

def inventory_counts(obs):
    inv=obs.get('inventory')
    if not isinstance(inv,dict):raise ValueError('Inventory scorer unavailable')
    return {k.removeprefix('minecraft:'):int(np.asarray(v).sum()) for k,v in inv.items()}

def milestone_counts(inv):
    return dict(log=sum(inv.get(k,0) for k in LOG_ITEMS),planks=sum(v for k,v in inv.items() if k=='planks' or k.endswith('_planks')),wooden_pickaxe=inv.get('wooden_pickaxe',0),diamond=inv.get('diamond',0))

@torch.no_grad()
def evaluate_episode(env,vpt,mapper,transformer,seed,budget,path):
    import cv2,hashlib
    env.seed(seed);obs=env.reset();obs=obs[0] if isinstance(obs,tuple) else obs
    initial=inventory_counts(obs)
    if any(v for v in initial.values()):raise ValueError('Evaluation requires empty starting inventory')
    state=vpt.initial_state(1);generator=torch.Generator(device='cuda:0').manual_seed(seed+901)
    hits={k:None for k in milestone_counts(initial)};started=time.monotonic()
    video=cv2.VideoWriter(str(path.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,360))
    if not video.isOpened():raise RuntimeError('Video writer unavailable')
    initial_hash=hashlib.sha256(obs['pov'].tobytes()).hexdigest()
    try:
        with path.open('x') as log:
            for step in range(budget):
                # Explicitly pass only pixels to the policy; inventory is only read below.
                image=cv2.resize(obs['pov'],(128,128),interpolation=cv2.INTER_LINEAR)
                (h,_),state=vpt.net({'img':torch.from_numpy(image.copy())[None,None].cuda()},state,context={'first':torch.tensor([[step==0]],device='cuda')})
                pd=vpt.pi_head(h);sampled={k:torch.multinomial(pd[k].reshape(-1,pd[k].shape[-1]).softmax(-1),1,generator=generator).cpu().numpy().reshape(1,1) for k in ('buttons','camera')}
                mapped=transformer.policy2env(mapper.to_factored(sampled))
                action={k:(np.asarray(v,dtype=np.float32).reshape(2) if k=='camera' else int(np.asarray(v).reshape(-1)[0])) for k,v in mapped.items()}
                if step%2==0:video.write(cv2.cvtColor(obs['pov'],cv2.COLOR_RGB2BGR))
                result=env.step(action)
                if len(result)==5:obs,reward,terminated,truncated,info=result;done=terminated or truncated
                else:obs,reward,done,info=result
                counts=milestone_counts(inventory_counts(obs))
                for key,count in counts.items():
                    if count>0 and hits[key] is None:hits[key]=step+1
                log.write(json.dumps(dict(step=step+1,buttons=int(sampled['buttons'][0,0]),camera=int(sampled['camera'][0,0]),milestones=counts,done=bool(done)))+'\n')
                if (step+1)%1000==0:print('STEP',seed,step+1,hits,flush=True)
                if done:break
    finally:video.release()
    return dict(status='ok',world_seed=seed,steps=step+1,first_hit=hits,initial_pixels_sha256=initial_hash,wall_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=19200);a=p.parse_args()
    protocol=json.loads(Path(a.protocol).read_text());out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    for e in protocol['assets'].values():
        if sha256(e['path'])!=e['sha256']:raise ValueError('Evaluation asset changed')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    # Load installed MineRL before the inference-only dependency shims are added.
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.env.malmo import InstanceManager
    from .vpt_rl import load_vpt
    assets=protocol['assets'];vpt,mapper,transformer=load_vpt(Path(assets['model']['path']),Path(assets['base']['path']),model_sha256=assets['model']['sha256'],weights_sha256=assets['base']['sha256'],device='cuda:0')
    base={k:v.detach().cpu().clone() for k,v in vpt.pi_head.state_dict().items()}
    heads={'baseline':base}
    for seed in (0,1,2):
        ck=torch.load(assets[f'rl_{seed}']['path'],map_location='cpu',weights_only=True)
        if ck['base_weights_sha256']!=assets['base']['sha256']:raise ValueError('Wrong RL backbone')
        heads[f'rl_{seed}']=ck['pi_head']
    from agent import ENV_KWARGS
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.env.malmo import InstanceManager
    InstanceManager.configure_malmo_base_port(a.port)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[]
    for seed in protocol['world_seeds']:
        order=list(heads);random.Random(seed).shuffle(order)
        for arm in order:
            env=None
            try:
                vpt.pi_head.load_state_dict(heads[arm],strict=True)
                env=HumanSurvival(**ENV_KWARGS).make()
                print('EPISODE',arm,seed,flush=True)
                r=evaluate_episode(env,vpt,mapper,transformer,seed,protocol['max_steps'],out/f'{arm}_{seed}.jsonl')
            except Exception as exc:
                r=dict(status='failed',world_seed=seed,error=f'{type(exc).__name__}: {exc}')
            finally:
                if env is not None:env.close()
            r['arm']=arm;records.append(r)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print('RESULT',json.dumps(r),flush=True)
            if r['status']=='failed':raise RuntimeError(r['error'])
    summaries={}
    for arm in heads:
        rr=[r for r in records if r['arm']==arm]
        summaries[arm]={task:dict(successes=sum(r['first_hit'][task] is not None for r in rr),trials=len(rr)) for task in ('log','planks','wooden_pickaxe','diamond')}
    (out/'summary.json').write_text(json.dumps(dict(status='complete',arms=summaries,real_environment=True,selection_or_training_feedback=False),indent=2))
if __name__=='__main__':main()
