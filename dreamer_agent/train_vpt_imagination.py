"""V7 Q4 low-level imagination RL (B1) in the frozen Dreamerv4-MC world model (PLAN_V7 §5.1, §7).

One world model + frozen VPT-3x backbone per GPU (torchrun). Each iteration: sample a
real root (train 'fit' session, skill-segment start), prefill the aligned world adapter
with 16 real frames and recorded actions, warm the VPT state on 128 real frames, roll
out the skill-conditioned policy for H steps, score with the Q1 reward rule (local
first completion of the root skill), then update
  policy: PMPO (alpha=0.5) + beta_KL * KL(pi || prior) + lambda_BC * BC(real skill frames)
  value : symexp twohot on lambda-returns (cont = 1 - local_done), EMA target.
Gradients are all-reduced across ranks. Every `prior_every`-th rollout uses the fixed
prior (no update) to monitor the reward-rate ratio rho (§7.3). Real evaluation data is
never used.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .provenance import sha256
from .reward_rule import RuleParams, detect
from .train_vpt_heads import prefetch
from .train_vpt_skill_bc import SkillSampler, joint_logprob
from .v7_data import CACHE, MILESTONES, SKILL_MAX_FRAMES, SKILLS, load_cache
from .vpt_features import generated_to_rgb, load_vpt_policy, to_vpt_input
from .vpt_heads import SignalHeads, SkillPolicy, TwoHotValue, action_kl, value_input

ROOT = Path(__file__).resolve().parents[1]
VPT_MODEL = ROOT / 'checkpoints/foundation-model-3x.model'
VPT_WEIGHTS = ROOT / 'checkpoints/bc-early-game-3x.weights'


def load_roots(rank, world):
    paths = sorted((CACHE / 'roots').glob('*.npz'))
    by_skill = {}
    for i, p in enumerate(paths):
        if i % world != rank:
            continue
        skill = p.stem.rsplit('_', 1)[0]
        with np.load(p) as z:
            meta = json.loads(str(z['meta']))
        by_skill.setdefault(meta['skill'], []).append(p)
    return by_skill


def root_tensors(path, codec, device):
    with np.load(path) as z:
        vpt_frames, wm_frames = z['vpt_frames'], z['wm_frames']
        rows, meta = json.loads(str(z['rows'])), json.loads(str(z['meta']))
    tokens = [codec.get_action_index_from_actiondict(codec.json_action_to_env_action(r, hotbar=True)[0], include_gui=True) for r in rows]
    pixels = torch.from_numpy(wm_frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    pixels = torch.nn.functional.pad(pixels, (0, 0, 12, 12))[None].to(device)
    return vpt_frames, pixels, torch.tensor(tokens, dtype=torch.long, device=device), meta


def lambda_returns(rewards, values, cont, bootstrap, gamma, lam):
    """rewards/cont [H], values [H+1] (values[H] replaced by bootstrap)."""
    H = len(rewards)
    out = torch.zeros(H, device=rewards.device)
    nxt = bootstrap
    for t in reversed(range(H)):
        v_next = values[t + 1] if t + 1 < H else bootstrap
        nxt = rewards[t] + gamma * cont[t] * ((1 - lam) * v_next + lam * nxt)
        out[t] = nxt
    return out


def kl_to_prior(logp, prior):
    return action_kl(logp, prior)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--rule', type=Path, required=True)
    parser.add_argument('--skill-policy', type=Path, default=None, help='Q3 checkpoint; prior and init')
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--sampling-steps', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.9995)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta-kl', type=float, default=0.3)
    parser.add_argument('--lambda-bc', type=float, default=0.3)
    parser.add_argument('--bc-batch', type=int, default=256)
    parser.add_argument('--bc-files', type=int, default=300)
    parser.add_argument('--deadzone', type=float, default=0.01)
    parser.add_argument('--lr-policy', type=float, default=1e-5)
    parser.add_argument('--lr-film', type=float, default=3e-4)
    parser.add_argument('--lr-value', type=float, default=1e-4)
    parser.add_argument('--prior-every', type=int, default=8)
    parser.add_argument('--skills', default=','.join(MILESTONES))
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path, default=None, help='skill_policy.pt: continue it, weights and optimizers')
    args = parser.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    torch.manual_seed(args.seed * 1000 + rank)
    rng = random.Random(args.seed * 1000 + rank)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()

    from .throughput_probe import load_world
    from .vpt_rl import VPTToWorldAction
    wm, spec, codec = load_world(args.sampling_steps, aligned=True)
    vpt, mapper, transformer, _ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL),
                                                  weights_sha256=sha256(VPT_WEIGHTS), device=device)
    convert = VPTToWorldAction(mapper, transformer, codec, spec, device)
    blob = torch.load(args.heads, map_location='cpu', weights_only=False)
    heads = SignalHeads(blob['members']).to(device).eval()
    heads.load_state_dict(blob['state'])
    rule = RuleParams(**json.loads(args.rule.read_text()))
    policy = SkillPolicy(vpt.pi_head).to(device)
    if args.skill_policy:
        sp = torch.load(args.skill_policy, map_location='cpu', weights_only=False)
        for name in ('pi_head', 'gamma', 'beta'):
            getattr(policy, name).load_state_dict(sp[name])
    prior = SkillPolicy(vpt.pi_head).to(device)
    prior.load_state_dict(policy.state_dict())
    prior.requires_grad_(False).eval()
    value = TwoHotValue().to(device)
    target = TwoHotValue().to(device)
    target.load_state_dict(value.state_dict())
    target.requires_grad_(False)
    for module in (policy, value):
        for p in module.parameters():
            dist.broadcast(p.data, 0)
    opt_pi = torch.optim.AdamW([dict(params=policy.pi_head.parameters(), lr=args.lr_policy),
                                dict(params=list(policy.gamma.parameters()) + list(policy.beta.parameters()), lr=args.lr_film)], weight_decay=0)
    opt_v = torch.optim.AdamW(value.parameters(), lr=args.lr_value, weight_decay=0)
    start_it = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.pi_head.load_state_dict(ckpt['pi_head'])
        policy.gamma.load_state_dict(ckpt['gamma'])
        policy.beta.load_state_dict(ckpt['beta'])
        value.load_state_dict(ckpt['value'])
        target.load_state_dict(ckpt['target'])
        opt_pi.load_state_dict(ckpt['opt_pi'])
        opt_v.load_state_dict(ckpt['opt_v'])
        start_it = int(ckpt['it']) + 1

    skills = [s for s in args.skills.split(',') if s]
    roots = {s: v for s, v in load_roots(rank, world).items() if s in skills}
    if not roots:
        raise RuntimeError(f'rank {rank}: no roots for {skills}')
    bc_files = [f for i, f in enumerate(load_cache({'fit'})) if i % world == rank][:args.bc_files]
    bc_stream = prefetch(SkillSampler(bc_files, args.seed * 1000 + rank), args.bc_batch, device, depth=2)

    config = dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, world=world,
                  heads_sha256=sha256(args.heads), rule=rule.to_dict(),
                  skill_policy_sha256=sha256(args.skill_policy) if args.skill_policy else None,
                  prior='skill_policy' if args.skill_policy else 'bc-early-game-3x pi_head', prior_weights_sha256=sha256(VPT_WEIGHTS),
                  cache_selection_sha256=sha256(CACHE / 'selection.jsonl'),
                  roots_per_skill_rank0={s: len(v) for s, v in roots.items()},
                  code_sha256={p: sha256(Path(__file__).parent / p) for p in
                               ('train_vpt_imagination.py', 'world_v7.py', 'vpt_heads.py', 'reward_rule.py', 'vpt_features.py', 'v7_data.py')},
                  caveats=['reward rule calibrated on real frames only (Q2 generated-domain annotation pending)',
                           'generated GUI frames carry no cursor overlay while VPT saw cursors on real GUI frames',
                           'WM camera gain ~20% below real (Q0)', 'no K risk branches yet'])
    if rank == 0:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2))
        (args.output / 'videos').mkdir(exist_ok=True)
    log = (args.output / f'metrics_rank{rank}.jsonl').open('a')
    started, transitions = time.time(), 0
    stats = {s: dict(policy=[0, 0], prior=[0, 0]) for s in skills}
    if args.resume:
        stats = ckpt['stats']

    for it in range(start_it, args.iterations + 1):
        use_prior = it % args.prior_every == 0
        skill_name = rng.choice(list(roots))
        path = rng.choice(roots[skill_name])
        k = SKILLS.index(skill_name)
        vpt_frames, pixels, tokens, meta = root_tensors(path, codec, device)
        t0 = time.time()
        with torch.no_grad():
            wm.reset(pixels, tokens, seed=rng.randrange(2 ** 31))
            state = vpt.initial_state(1)
            first = torch.zeros(1, len(vpt_frames), dtype=torch.bool, device=device)
            first[0, 0] = True
            (hseq, _), state = vpt.net({'img': torch.from_numpy(vpt_frames)[None].to(device)}, state, context={'first': first})
            hs = [hseq[0, -2].float(), hseq[0, -1].float()]
            actions, frames = [], []
            skill_t = torch.tensor([k], device=device)
            actor = prior if use_prior else policy
            for t in range(args.horizon):
                logp = actor(hs[-1][None], skill_t)
                a = torch.stack([torch.multinomial(logp[key][0].exp(), 1)[0] for key in ('buttons', 'camera')])
                actions.append(a)
                wm_tokens = torch.as_tensor(spec.to_tokens(convert(a).cpu().numpy()), device=device, dtype=torch.long)[None]
                _, frame = wm.step(wm_tokens)
                rgb = generated_to_rgb(frame)
                if t % 4 == 0:
                    frames.append(rgb[12:372:2, ::2])
                img = torch.from_numpy(to_vpt_input(rgb).copy())[None, None].to(device)
                (h, _), state = vpt.net({'img': img}, state, context={'first': torch.tensor([[False]], device=device)})
                hs.append(h[0, 0].float())
            H = torch.stack(hs)  # [horizon + 2]: h_{-1}, h_0 .. h_H
            sig = heads.signals(H[1:], H[:-1])
            signals = {key: v.cpu().numpy() for key, v in sig.items()}
            det = detect(signals, rule, skill_name)
            # signals index 0 is the root frame; a detection at index j >= 1 rewards the action taken at j-1.
            det = [j for j in det if j >= 1]
            done_at = det[0] if det else None
        rewards = torch.zeros(args.horizon, device=device)
        cont = torch.ones(args.horizon, device=device)
        if done_at is not None:
            rewards[done_at - 1] = 1.0
            cont[done_at - 1:] = 0.0
        stats[skill_name]['prior' if use_prior else 'policy'][0] += int(done_at is not None)
        stats[skill_name]['prior' if use_prior else 'policy'][1] += 1
        transitions += args.horizon
        record = dict(it=it, rank=rank, skill=skill_name, root=path.name, prior=use_prior, reward=float(rewards.sum()),
                      done_at=done_at, rollout_sec=round(time.time() - t0, 1))
        if done_at is not None and rank == 0 and len(list((args.output / 'videos').glob('*.npz'))) < 500:
            np.savez_compressed(args.output / 'videos' / f'it{it}_{skill_name}_{"prior" if use_prior else "policy"}.npz',
                                frames=np.stack(frames), done_at=done_at)
        if not use_prior:
            states = H[1:]  # s_0 .. s_H
            skill_vec = torch.full((len(states),), k, device=device)
            progress = torch.arange(len(states), device=device).float() / SKILL_MAX_FRAMES[skill_name]
            vin = value_input(states, skill_vec, progress.clamp(max=1), torch.ones(len(states), device=device))
            with torch.no_grad():
                v_target = target.value(vin)
                bootstrap = v_target[-1].clamp(0, 1)
                ret = lambda_returns(rewards, v_target, cont, bootstrap, args.gamma, args.lam)
                adv = ret - value.value(vin[:-1])
            act = torch.stack(actions)
            logp = policy(states[:-1], skill_vec[:-1])
            lp = joint_logprob(logp, act[:, 0], act[:, 1])
            pos, neg = adv > args.deadzone, adv < -args.deadzone
            loss_pg = torch.zeros((), device=device)
            if pos.any():
                loss_pg = loss_pg - args.alpha * lp[pos].mean()
            if neg.any():
                loss_pg = loss_pg + (1 - args.alpha) * lp[neg].mean()
            with torch.no_grad():
                pr = prior(states[:-1], skill_vec[:-1])
            kl = kl_to_prior(logp, pr).mean()
            b = next(bc_stream)
            valid = ~b['null']
            bc = -(joint_logprob(policy(b['h'], b['skill']), b['buttons'], b['camera']) * valid).sum() / valid.sum().clamp(min=1)
            loss_pi = loss_pg + args.beta_kl * kl + args.lambda_bc * bc
            loss_v = value.loss(vin[:-1], ret).mean()
            opt_pi.zero_grad(set_to_none=True)
            opt_v.zero_grad(set_to_none=True)
            (loss_pi + loss_v).backward()
            for module in (policy, value):
                for p in module.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
            opt_pi.step()
            opt_v.step()
            with torch.no_grad():
                for p, q in zip(value.parameters(), target.parameters()):
                    q.lerp_(p, 0.02)
            record.update(kl=float(kl), bc=float(bc), loss_pg=float(loss_pg), loss_v=float(loss_v), adv_pos=int(pos.sum()),
                          adv_neg=int(neg.sum()), value_mean=float(v_target.mean()))
        else:
            # Keep ranks in lock-step: prior rollouts contribute zero gradients.
            opt_pi.zero_grad(set_to_none=True)
            opt_v.zero_grad(set_to_none=True)
            for module in (policy, value):
                for p in module.parameters():
                    g = torch.zeros_like(p)
                    dist.all_reduce(g, op=dist.ReduceOp.AVG)
        log.write(json.dumps(record) + '\n')
        log.flush()
        if it % 50 == 0 and rank == 0:
            elapsed = time.time() - started
            rates = {s: (round(v['policy'][0] / max(1, v['policy'][1]), 3), round(v['prior'][0] / max(1, v['prior'][1]), 3),
                         v['policy'][1]) for s, v in stats.items() if v['policy'][1]}
            print(f'it {it} tps/rank {transitions / elapsed:.2f} rates(policy,prior,n) {rates}', flush=True)
        if it % args.save_every == 0 and rank == 0:
            torch.save(dict(pi_head=policy.pi_head.state_dict(), gamma=policy.gamma.state_dict(), beta=policy.beta.state_dict(),
                            value=value.state_dict(), target=target.state_dict(), opt_pi=opt_pi.state_dict(), opt_v=opt_v.state_dict(),
                            it=it, stats=stats, config=config, skills=SKILLS), args.output / 'skill_policy.pt')
            (args.output / 'stats.json').write_text(json.dumps(dict(it=it, stats=stats, transitions_rank0=transitions), indent=1))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
