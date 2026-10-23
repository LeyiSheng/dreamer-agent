"""Explicitly authorized, bounded VPT proxy RL. No supervised updates.

Not the evidence-gated P2 runner. Existing reward remains uncalibrated.
"""
import argparse,json,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .provenance import sha256
from .proxy_rl_pilot import ProxyPilotTrainer
from .imagination import RLConfig


class DirectTrainer(ProxyPilotTrainer):
    def collect(self,*args,**kwargs):
        trajectory,metrics=super().collect(*args,**kwargs)
        # Finite-horizon proxy value depends on time remaining, not a success streak.
        n=len(trajectory.history)
        trajectory.confirmation=torch.arange(n,device=trajectory.history.device)/n
        trajectory.next_confirmation=torch.arange(1,n+1,device=trajectory.history.device)/n
        return trajectory,metrics


def save_checkpoint(path,trainer,protocol):
    state=dict(actor=trainer.actor.state_dict(),prior=trainer.prior.state_dict(),value=trainer.value.state_dict(),
        optimizer=trainer.optimizer.state_dict(),generator=trainer.generator.get_state(),update=trainer.version,
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],protocol=protocol)
    tmp=Path(str(path)+'.tmp');torch.save(state,tmp);tmp.replace(path)


def restore_checkpoint(path,trainer,protocol):
    ck=torch.load(path,map_location=next(trainer.actor.parameters()).device,weights_only=True)
    if ck['protocol']!=protocol:raise ValueError('Resume protocol/assets mismatch')
    for key in ('actor','prior','value'):getattr(trainer,key).load_state_dict(ck[key],strict=True)
    trainer.optimizer.load_state_dict(ck['optimizer']);trainer.generator.set_state(ck['generator'].cpu())
    trainer.version=ck['update'];torch.set_rng_state(ck['cpu_rng'].cpu())
    if ck['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])


def root_index(update,seed,count):
    if count<1:raise ValueError('Empty roots')
    order=list(range(count));random.Random(seed+update//count).shuffle(order)
    return order[update%count]


def load_roots(config,backend,out):
    import cv2
    from .contracts import Episode
    from .imagination_backend import OfflineRoot
    from .rl_roots import tensor_hash
    rb=backend.reward_backend;codec=backend.vpt_to_local.codec;roots=[]
    for index,entry in enumerate(config['roots']):
        for kind in ('video','actions'):
            if sha256(entry[kind]['path'])!=entry[kind]['sha256']:raise ValueError('Root source changed')
        cap=cv2.VideoCapture(entry['video']['path']);bgr=None
        try:
            for _ in range(entry['frame']+1):
                ok,bgr=cap.read()
                if not ok:raise ValueError('Root decode failed')
        finally:cap.release()
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        pixels=torch.nn.functional.pad(torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/127.5-1,(0,0,12,12))[None,None]
        row=json.loads(Path(entry['actions']['path']).read_text().splitlines()[entry['frame']])
        tokens=torch.tensor(codec.get_action_index_from_actiondict(codec.json_action_to_env_action(row,hotbar=True)[0],include_gui=True),device='cuda',dtype=torch.long)[None]
        features=rb.encode_pixels(pixels)[None];past=torch.zeros(1,sum(rb.bc.action_sizes),device='cuda')
        ep=Episode(episode_id=entry['stem']+':'+str(entry['frame']),source_episode_id=entry['source_session'],split='train',origin='offline')
        manifest=dict(schema_version=1,episode=vars(ep),frame=entry['frame'],task_start_frame=entry['frame'],
            feature_contract_sha256=sha256('dreamer_agent/rl_pixels.py'),alignment_evidence_sha256=sha256(out/'integration/alignment_contract.json'),
            sources=[entry['video'],entry['actions']],tensors={k:tensor_hash(v) for k,v in dict(pixels=pixels,frame_actions=tokens,features=features,past_actions=past).items()})
        path=out/f'root_{index}.json'
        if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Root changed on resume')
        path.write_text(json.dumps(manifest,indent=2))
        roots.append(OfflineRoot(ep,pixels,tokens,features,dict(manifest_path=str(path.resolve()),manifest_sha256=sha256(path)),past))
    return roots


def run(config_path,out,resume=False):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    if cfg.get('scope')!='vpt_direct_proxy_rl' or cfg.get('user_authorized') is not True:raise ValueError('Explicit experiment protocol required')
    if cfg['updates']!=100 or cfg['seeds']!=[0,1,2] or len(cfg['roots'])!=8:raise ValueError('Outside fixed pilot budget')
    for v in cfg['artifacts'].values():
        if sha256(v['path'])!=v['sha256']:raise ValueError('Bound artifact changed: '+v['path'])
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    from .verify_vpt_rl_integration import build_backend
    from .vpt_rl import VPTReadout
    # Reconstruct frozen backend after interruption; do not restore half a rollout.
    integration=out/'integration'
    if integration.exists():
        index=1
        while (out/f'integration_restart{index}').exists():index+=1
        integration=out/f'integration_restart{index}'
    backend,vpt,_,reward_path,binding=build_backend(integration)
    roots=load_roots(cfg,backend,out)
    rl=RLConfig(horizon=16,learning_rate=3e-5,bc_weight=0.)
    protocol=dict(config_sha256=sha256(config_path),config=cfg,rl=asdict(rl),reward_calibrated=False,real_success_measured=False,
        value_auxiliary='elapsed fraction of artificial 16-step objective; not completion',supervised_updates=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start=time.monotonic()
    for seed in cfg['seeds']:
        sub=out/f'seed_{seed}';sub.mkdir(exist_ok=True)
        if (sub/'summary.json').exists():continue
        trainer=DirectTrainer(vpt,backend,rl,{0:.5},seed=seed,policy_readout=VPTReadout(vpt),history_dim=vpt.net.output_latent_size(),tasks=1)
        ck=sub/'latest.pt'
        if ck.exists():restore_checkpoint(ck,trainer,dict(protocol,seed=seed))
        baseline={k:v.detach().cpu().clone() for k,v in trainer.prior.state_dict().items()}
        with (sub/'metrics.jsonl').open('a') as log:
            while trainer.version<cfg['updates']:
                update=trainer.version;ri=root_index(update,seed,len(roots));world_seed=seed*100000+update
                tr,_=trainer.collect(roots[ri],0,seed=world_seed)
                metrics=trainer.update(tr,None)
                with torch.no_grad():
                    from torch.distributions import Categorical,kl_divergence
                    pi=trainer.actor(tr.history,tr.tasks);prior=trainer.prior(tr.history,tr.tasks)
                    kl=float(sum(kl_divergence(Categorical(logits=a),Categorical(logits=b)).mean() for a,b in zip(pi,prior)))
                if not np.isfinite(kl) or kl>1.:raise RuntimeError('Pilot KL limit exceeded; stopped without accepting update')
                if any(not torch.isfinite(p).all() for p in [*trainer.actor.parameters(),*trainer.value.parameters()]):raise RuntimeError('Nonfinite parameter')
                if any(p.grad is not None for net in trainer.frozen for p in net.parameters()):raise RuntimeError('Frozen gradient leak')
                metrics.update(seed=seed,root_index=ri,post_update_kl=kl,elapsed_seconds=time.monotonic()-start,
                    actor_max_abs_change=max(float((v.detach().cpu()-baseline[k]).abs().max()) for k,v in trainer.actor.state_dict().items()))
                save_checkpoint(ck,trainer,dict(protocol,seed=seed))
                log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush();print(json.dumps(metrics,allow_nan=False),flush=True)
        torch.save(dict(pi_head=trainer.actor.head.state_dict(),base_weights_sha256=binding['weights_sha256'],protocol=dict(protocol,seed=seed)),sub/'vpt_head.pt')
        (sub/'summary.json').write_text(json.dumps(dict(status='complete',updates=trainer.version,supervised_updates=0,world_steps=trainer.version*16,real_success_measured=False),indent=2))
        del trainer
    if sha256('checkpoints/bc-early-game-3x.weights')!=binding['weights_sha256']:raise RuntimeError('Original VPT changed')
    (out/'summary.json').write_text(json.dumps(dict(status='complete',seeds=cfg['seeds'],updates_per_seed=100,world_steps=4800,supervised_updates=0,calibration_passed=False,real_success_measured=False),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.output,a.resume)
if __name__=='__main__':main()
