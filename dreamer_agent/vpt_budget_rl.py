"""User-authorized eight-hour VPT proxy RL with diverse roots and streaming readout.

No calibration claim, no supervised updates, no learned hierarchy or real-world
feedback. Deadline, provenance, numerical equivalence and KL remain explicit.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from .contracts import Episode
from .imagination import RLConfig
from .imagination_backend import OfflineRoot
from .provenance import sha256
from .rl_contract import HistoryRewardContract, StreamingReadout, module_sha256
from .rl_roots import tensor_hash
from .vpt_direct_rl import DirectTrainer, root_index, save_checkpoint, restore_checkpoint


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def next_seed(versions):
    """Fair interleaving, including when recovery checkpoints have unequal ages."""
    return min(versions, key=lambda seed: (versions[seed], seed))


def load_budget(path, seconds, now):
    """Recovery preserves the original wall-clock deadline, never grants more time."""
    path = Path(path)
    if path.exists():
        budget = json.loads(path.read_text())
        if budget['budget_seconds'] != seconds or budget['deadline_timestamp']-budget['start_timestamp'] != seconds:
            raise ValueError('Budget changed or malformed')
        return budget
    budget = dict(started_at=iso(now),deadline_at=iso(now+seconds),
        start_timestamp=now,deadline_timestamp=now+seconds,budget_seconds=seconds,
        note='Model/data preparation excluded; stop after the first update crossing the deadline')
    atomic_json(path,budget)
    return budget


def validate_config(cfg):
    if cfg.get('scope') != 'vpt_proxy_rl_time_budget_v2' or cfg.get('user_authorized') is not True:
        raise ValueError('Explicit time-budget proxy experiment authorization required')
    if cfg.get('budget_seconds') != 28800 or cfg.get('seeds') != [0, 1, 2]:
        raise ValueError('Expected fixed eight-hour, three-seed budget')
    if cfg.get('horizon') != 16 or cfg.get('bc_weight') != 0 or cfg.get('learning_rate') != 3e-5:
        raise ValueError('Undeclared change to the proxy objective')
    if cfg.get('kl_limit') != 1. or cfg.get('checkpoint_every') != 20:
        raise ValueError('Undeclared stopping/checkpoint rule')


def verify_membership(manifest):
    entry = manifest['audit']
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('Source audit changed')
    selected = {r['stem']: r for r in manifest['roots']}
    matched = set()
    held_out_groups = set()
    with Path(entry['path']).open() as f:
        for line in f:
            r = json.loads(line)
            group = r['source_session'].rsplit('-', 1)[-1]
            if r['split'] != 'train': held_out_groups.add(group)
            if r['stem'] not in selected: continue
            s = selected[r['stem']]
            if r['split'] != 'train' or r['status'] != 'eligible_pending_full_decode':
                raise ValueError('Root outside eligible train set')
            if r['source_session'] != s['source_session'] or r['actions_sha256'] != s['actions']['sha256']:
                raise ValueError('Root provenance differs from audit')
            if r['video_path'] != s['video']['path'] or r['actions_path'] != s['actions']['path']:
                raise ValueError('Root source path differs from audit')
            matched.add(r['stem'])
    if matched != set(selected) or any(r['session_group'] in held_out_groups for r in selected.values()):
        raise ValueError('Missing source or held-out session collision')
    if len(manifest['roots']) != 512 or len({r['session_group'] for r in manifest['roots']}) != 256:
        raise ValueError('Expected 512 roots from 256 conservative session groups')
    if len({(r['stem'],r['frame']) for r in manifest['roots']}) != 512:
        raise ValueError('Duplicated roots')
    for r in manifest['roots']:
        if r['source_split'] != 'train' or r['source_origin'] != 'offline':
            raise ValueError('Evaluation inputs are not allowed')


class RootBank:
    def __init__(self, manifest, backend, output, alignment):
        self.entries, self.backend = manifest['roots'], backend
        self.output, self.alignment = Path(output), alignment
        self.output.mkdir(exist_ok=True)
        self.features = {}

    def get(self, index):
        entry = self.entries[index]
        rb = self.backend.reward_backend
        if sha256(entry['image']['path']) != entry['image']['sha256']:
            raise ValueError('Extracted root image changed')
        with np.load(entry['image']['path'], allow_pickle=False) as data:
            rgb = data['rgb']
        if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('Bad source root image')
        pixels = torch.nn.functional.pad(torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/127.5-1,
                                         (0,0,12,12))[None,None]
        codec = self.backend.vpt_to_local.codec
        tokens = torch.tensor(codec.get_action_index_from_actiondict(codec.json_action_to_env_action(entry['action_row'],hotbar=True)[0],include_gui=True),device='cuda',dtype=torch.long)[None]
        if index not in self.features:
            self.features[index] = rb.encode_pixels(pixels)[None].cpu()
        features = self.features[index].cuda()
        past = torch.zeros(1,sum(rb.bc.action_sizes),device='cuda')
        ep = Episode(entry['stem']+':'+str(entry['frame']),entry['source_session'],'train','offline')
        tensors = dict(pixels=pixels,frame_actions=tokens,features=features,past_actions=past)
        record = dict(schema_version=1,episode=vars(ep),frame=entry['frame'],task_start_frame=entry['frame'],
            feature_contract_sha256=sha256('dreamer_agent/rl_pixels.py'),alignment_evidence_sha256=sha256(self.alignment),
            sources=[entry['video'],entry['actions'],entry['image']],
            tensors={k:tensor_hash(v) for k,v in tensors.items()},role=entry['role'])
        path = self.output/f'root_{index}.json'
        if path.exists():
            if json.loads(path.read_text()) != record: raise ValueError('Root changed on resume')
        else: atomic_json(path,record)
        return OfflineRoot(ep,pixels,tokens,features,dict(manifest_path=str(path.resolve()),manifest_sha256=sha256(path)),past)


@torch.no_grad()
def enable_streaming(backend, vpt, root, output):
    """Measured migration of legacy inference, explicitly NOT training-metadata certification."""
    rb = backend.reward_backend
    bc_path = Path('outputs/p1b_task_bc_v1/seed_0/bc.pt')
    reward_path = Path('outputs/p2_reward_screen_20260918_v2/reward_snapshot.pt')
    data_path = Path('outputs/p1b_event_coverage_20260914/coverage.json')
    bc = torch.load(bc_path,map_location='cpu',weights_only=True)['metadata']
    reward = torch.load(reward_path,map_location='cpu',weights_only=True)['metadata']
    if reward['bc_sha256'] != sha256(bc_path) or bc['context'] != 32:
        raise ValueError('Legacy training provenance mismatch')
    if bc['action_spec_sha256'] != rb.world.action_spec.fingerprint or reward['extractor_sha256'] != bc['feature_spec']['extractor_sha256']:
        raise ValueError('Legacy action/feature provenance mismatch')
    tasks = tuple(json.loads(data_path.read_text())['tasks'])
    if tasks[0] != 'obtain_log' or len(tasks) != 5: raise ValueError('Unexpected legacy task order')
    contract = HistoryRewardContract(32,tasks,tasks,module_sha256(rb.bc),module_sha256(rb.reward),
        sha256('dreamer_agent/rl_pixels.py'),rb.world.action_spec.fingerprint)
    # Do not populate rb.contract or fabricate reward training metadata. The strict
    # formal contract path stays unavailable to these old uncalibrated weights.
    actions, histories, scores = [], [], []
    history = backend.reset(root,0,987654)
    from .vpt_rl import VPTReadout
    actor = VPTReadout(vpt).eval()
    for _ in range(16):
        action = torch.stack([x.argmax(-1) for x in actor(history,torch.tensor(0,device=history.device))])
        result = backend.step(action)
        actions.append(action);histories.append(result.history.clone());scores.append(result.completion_probability)
        history = result.history
    del actor
    rb.readout = StreamingReadout(rb.bc,rb.reward,contract)
    backend.reset(root,0,987654)
    history_error, score_error = 0.,0.
    for action, reference, score in zip(actions,histories,scores):
        result = backend.step(action)
        history_error = max(history_error,float((result.history-reference).abs().max()))
        score_error = max(score_error,abs(result.completion_probability-score))
    if history_error > 1e-4 or score_error > 1e-5:
        raise RuntimeError('Streaming real-model replay is not numerically equivalent')
    report = dict(kind='legacy_inference_equivalence',world_steps=32,optimizer_updates=0,
        policy_max_abs_difference=history_error,reward_max_abs_difference=score_error,
        contract=asdict(contract),legacy_policy_metadata=bc,legacy_reward_metadata=reward,
        task_order_evidence=dict(path=str(data_path),sha256=sha256(data_path)),
        original_feature_manifest_available=False,
        provenance_limitation='Original P1b feature manifest is absent; task order comes from retained coverage and is not newly certified.',
        calibration_passed=False,training_metadata_migrated=False,formal_contract_accepted=False)
    atomic_json(Path(output)/'streaming_preflight.json',report)
    print('STREAMING_PREFLIGHT',json.dumps({k:v for k,v in report.items() if k not in ('contract','legacy_policy_metadata','legacy_reward_metadata')}),flush=True)


def checkpoint_all(trainers, output, protocol):
    if shutil.disk_usage(output).free < 5*1024**3:
        raise RuntimeError('Less than 5 GiB free; preserve existing checkpoints')
    for seed, trainer in trainers.items():
        save_checkpoint(Path(output)/f'seed_{seed}'/'latest.pt',trainer,dict(protocol,seed=seed))


def run(config_path, output, resume=False):
    from .verify_vpt_rl_integration import build_backend
    from .vpt_rl import VPTReadout
    cfg = json.loads(Path(config_path).read_text());validate_config(cfg)
    out = Path(output);out.mkdir(parents=True,exist_ok=resume)
    lock = (out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(out/'owner.json',dict(pid=os.getpid()))
    for artifact in cfg['artifacts'].values():
        if sha256(artifact['path']) != artifact['sha256']:
            raise ValueError('Bound artifact changed: '+artifact['path'])
    manifest = json.loads(Path(cfg['roots_manifest']).read_text())
    verify_membership(manifest)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.65)
    # Every process reconstruction receives a distinct diagnostic directory.
    attempt = len(list(out.glob('integration_*')))
    integration = out/f'integration_{attempt}'
    backend,vpt,_,_,binding = build_backend(integration)
    bank = RootBank(manifest,backend,out/'roots',integration/'alignment_contract.json')
    root = bank.get(0)
    enable_streaming(backend,vpt,root,out)
    rl = RLConfig(horizon=16,learning_rate=3e-5,bc_weight=0.)
    protocol = dict(config_sha256=sha256(config_path),config=cfg,rl=asdict(rl),
        reward_source='uncalibrated_terminal_proxy',reward_calibrated=False,supervised_updates=False,
        value_auxiliary='elapsed fraction of artificial 16-step objective',seed_interleaving='minimum update count',
        optimization='incremental frozen P1 history/reward readout; VPT backbone frozen')
    pp = out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol: raise ValueError('Protocol changed on resume')
    atomic_json(pp,protocol)
    trainers = {}
    for seed in cfg['seeds']:
        sub = out/f'seed_{seed}';sub.mkdir(exist_ok=True)
        trainer = DirectTrainer(vpt,backend,rl,{0:.5},seed=seed,policy_readout=VPTReadout(vpt),
            history_dim=vpt.net.output_latent_size(),tasks=1)
        if resume and (sub/'latest.pt').exists():
            restore_checkpoint(sub/'latest.pt',trainer,dict(protocol,seed=seed))
        trainers[seed] = trainer
    budget_path = out/'budget.json'
    budget = load_budget(budget_path,cfg['budget_seconds'],time.time())
    deadline = budget['deadline_timestamp']
    checkpoint_all(trainers,out,protocol)
    signal_stop = []
    import signal
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,f:signal_stop.append(s))
    visits = {seed:set() for seed in trainers}
    logfile = (out/f'metrics_run_{attempt}.jsonl').open('a')
    print('TRAINING_STARTED',json.dumps(dict(budget=budget,root_count=len(bank.entries),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
    while time.time() < deadline and not signal_stop:
        seed = next_seed({s:t.version for s,t in trainers.items()})
        trainer = trainers[seed]
        index = root_index(trainer.version,seed,len(bank.entries))
        root = bank.get(index)
        start = time.monotonic()
        tr,_ = trainer.collect(root,0,seed=seed*100000000+trainer.version)
        metrics = trainer.update(tr,None)
        with torch.no_grad():
            pi, prior = trainer.actor(tr.history,tr.tasks),trainer.prior(tr.history,tr.tasks)
            kl = float(sum(kl_divergence(Categorical(logits=a),Categorical(logits=b)).mean() for a,b in zip(pi,prior)))
        if not math.isfinite(kl) or kl > cfg['kl_limit']:
            raise RuntimeError('KL limit exceeded; failing update is not checkpointed')
        if any(not torch.isfinite(p).all() for p in [*trainer.actor.parameters(),*trainer.value.parameters()]):
            raise RuntimeError('Nonfinite trained parameter')
        if any(p.grad is not None for module in trainer.frozen for p in module.parameters()):
            raise RuntimeError('Frozen gradient leak')
        visits[seed].add(index)
        metrics.update(seed=seed,root_index=index,root_role=bank.entries[index]['role'],post_update_kl=kl,
            update_seconds=time.monotonic()-start,elapsed_seconds=time.time()-budget['start_timestamp'])
        logfile.write(json.dumps(metrics,allow_nan=False)+'\n');logfile.flush()
        versions = {s:t.version for s,t in trainers.items()}
        progress = dict(status='running',pid=os.getpid(),updated_at=iso(time.time()),budget=budget,updates=versions,
            world_steps=sum(versions.values())*16,roots_seen_this_run={s:len(v) for s,v in visits.items()},
            last_update=dict(seed=seed,root_index=index,kl=kl,reward=metrics['imagined_reward']),
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,calibration_passed=False)
        atomic_json(out/'progress.json',progress)
        print('UPDATE',json.dumps(dict(seed=seed,update=trainer.version,seconds=metrics['update_seconds'],kl=kl,
            reward=metrics['imagined_reward'],roots_seen=len(visits[seed]))),flush=True)
        if sum(versions.values()) % (cfg['checkpoint_every']*len(trainers)) == 0:
            checkpoint_all(trainers,out,protocol)
    logfile.close()
    checkpoint_all(trainers,out,protocol)
    for seed,trainer in trainers.items():
        path = out/f'seed_{seed}'/'vpt_head.pt'
        torch.save(dict(pi_head=trainer.actor.head.state_dict(),base_weights_sha256=binding['weights_sha256'],
            protocol=dict(protocol,seed=seed),update=trainer.version),path)
    summary = dict(status='interrupted' if signal_stop else 'budget_complete',budget=budget,
        finished_at=iso(time.time()),updates={s:t.version for s,t in trainers.items()},
        world_steps=sum(t.version for t in trainers.values())*16,supervised_updates=0,
        calibration_passed=False,real_success_measured=False,
        next_step='Fixed final-head real evaluation; never select checkpoints from these test outcomes')
    atomic_json(out/'summary.json',summary);atomic_json(out/'progress.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:run(a.config,a.output,a.resume)
    except Exception as exc:
        output=Path(a.output)
        owner=output/'owner.json'
        if owner.is_file() and json.loads(owner.read_text()).get('pid') == os.getpid():
            atomic_json(output/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',time=iso(time.time())))
        raise


if __name__ == '__main__': main()
