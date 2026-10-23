"""User-authorized bounded proxy pilot. Never marks P2 calibration as accepted.

Reward is ZERO except for the final step, where it is the uncalibrated reward
head score. It is an artificial finite-horizon objective, not task completion.
The strict calibrated training runner remains unchanged.
"""
import argparse,json,math,time
from pathlib import Path
from dataclasses import asdict
import torch
from .contracts import Episode
from .imagination import ImaginationTrainer,RLConfig,Trajectory
from .rl_data import load_bc_batch
from .provenance import sha256


class ProxyPilotTrainer(ImaginationTrainer):
    @torch.no_grad()
    def collect(self,root,task,*,seed):
        self._check_frozen()
        ep=root.episode
        if not isinstance(ep,Episode) or ep.origin!='offline' or ep.split!='train':
            raise ValueError('Only offline train roots in proxy pilot')
        if task!=0:raise ValueError('Single-task pilot only')
        h=self._history(self.backend.reset(root,task,seed))
        tid=torch.tensor(task,device=h.device,dtype=torch.long)
        rows=[];calls=0
        for i in range(self.config.horizon):
            scores=self.actor(h,tid)
            action=torch.stack([torch.multinomial(x.softmax(-1),1,generator=self.generator)[0] for x in scores])
            step=self.backend.step(action);calls+=1
            if step.uncertainty_cut or step.terminated or step.timeout:
                raise ValueError('Unexpected boundary in fixed-horizon proxy pilot; discard rollout')
            nh=self._history(step.history);p=float(step.completion_probability)
            if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Nonfinite or invalid proxy score')
            rows.append((h,nh,action,p));h=nh
        rewards=torch.zeros(len(rows),device=h.device);rewards[-1]=rows[-1][3]
        boundary=torch.zeros(len(rows),device=h.device,dtype=torch.bool);boundary[-1]=True
        zeros=torch.zeros(len(rows),device=h.device)
        # 'rewarded' is the existing zero-continuation mask. Here it means end of
        # artificial horizon, NEVER inferred environment/task completion.
        tr=Trajectory(torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),
            torch.stack([x[2] for x in rows]),tid.expand(len(rows)),rewards,
            boundary&False,boundary,boundary&False,self.version,ep.source_episode_id,calls,0,
            zeros,zeros,boundary.clone(),seed,zeros)
        self.last_scores=[r[3] for r in rows]
        return tr,dict(model_calls=calls,valid_steps=len(rows),uncertainty_cuts=0)

    def update(self,trajectory,offline):
        metrics=super().update(trajectory,offline)
        metrics.pop('objective_complete',None)
        metrics['reward_source']='uncalibrated_terminal_proxy'
        metrics['task_success_measured']=False
        metrics['horizon_ended']=True
        metrics['proxy_scores']=self.last_scores
        return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    auth=json.loads(a.authorization.read_text())
    if auth.get('scope')!='uncalibrated_proxy_rl_pilot' or auth.get('user_authorized') is not True:
        raise ValueError('Explicit proxy pilot authorization required')
    updates=auth['updates']
    if type(updates)is not int or not 1<=updates<=20 or auth['horizon']!=16:
        raise ValueError('Pilot exceeds authorized bounded budget')
    for entry in auth['artifacts'].values():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Pilot artifact changed')
    out=a.output;out.mkdir(exist_ok=False,parents=True)
    from .verify_vpt_rl_integration import build_backend
    from .vpt_rl import VPTReadout
    backend,vpt,root,reward_path,binding=build_backend(out/'integration')
    manifest=Path('outputs/p2_vpt_inputs_20260918/bc_manifest.json')
    batch=load_bc_batch(manifest,sha256(manifest),device='cuda:0')
    config=RLConfig(horizon=16,learning_rate=3e-5)
    trainer=ProxyPilotTrainer(vpt,backend,config,{0:.5},seed=0,data_binding=sha256(manifest),
        policy_readout=VPTReadout(vpt),history_dim=vpt.net.output_latent_size(),tasks=1)
    protocol=dict(kind='uncalibrated_proxy_rl_pilot',authorization=auth,config=asdict(config),
        code_sha256=sha256(__file__),calibration_passed=False,rl_ready=False,
        reward='only terminal sigmoid score at step16; zero objective bootstrap',
        source_episode_id=root.episode.source_episode_id,root_frame=0,
        actor_initialization_sha256=binding['weights_sha256'],reward_sha256=sha256(reward_path),
        limitations=['single training root','16-frame BC batch','no accepted semantic consistency','no real Minecraft evaluation'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    baseline={k:v.detach().cpu().clone() for k,v in trainer.actor.state_dict().items()}
    started=time.monotonic()
    with (out/'metrics.jsonl').open('x') as log:
        for i in range(updates):
            tr,_=trainer.collect(root,0,seed=100+i)
            metrics=trainer.update(tr,batch)
            with torch.no_grad():
                logits=trainer.actor(tr.history,tr.tasks);prior=trainer.prior(tr.history,tr.tasks)
                from torch.distributions import Categorical,kl_divergence
                metrics['post_update_kl']=float(sum(kl_divergence(Categorical(logits=x),Categorical(logits=y)).mean() for x,y in zip(logits,prior)))
                metrics['actor_max_abs_change']=max(float((v.detach().cpu()-baseline[k]).abs().max()) for k,v in trainer.actor.state_dict().items())
            if any(not torch.isfinite(p).all() for p in trainer.actor.parameters()):raise ValueError('Actor became nonfinite')
            if any(p.grad is not None for net in trainer.frozen for p in net.parameters()):raise ValueError('Frozen gradient leak')
            metrics['elapsed_seconds']=time.monotonic()-started
            log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
            torch.save(dict(actor=trainer.actor.state_dict(),value=trainer.value.state_dict(),optimizer=trainer.optimizer.state_dict(),
                generator=trainer.generator.get_state(),update=trainer.version,protocol=protocol),out/'latest.tmp')
            (out/'latest.tmp').replace(out/'latest.pt')
            print('UPDATE',json.dumps(metrics,allow_nan=False),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',updates=trainer.version,
        training_world_steps=16*trainer.version,real_success_measured=False,calibration_passed=False,
        original_weights_unchanged=sha256('checkpoints/bc-early-game-3x.weights')==binding['weights_sha256']),indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
