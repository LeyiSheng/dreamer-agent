"""V7 Q3: skill-conditioned BC on cached real VPT-3x features (PLAN_V7 §8 Q3).

Trainable copy of the bc-early-game-3x pi_head plus zero-initialised skill FiLM.
Loss per non-null frame: -log pi(buttons, camera | h, skill) (official VPT joint
log-prob), plus beta_kl * KL(pi(.|h, general) || pi_orig) on general frames so the
general mode stays at the original policy. Optional AWR weights (Q1-OFF) come from
dreamer_agent.offline_rl.

  python -m dreamer_agent.train_vpt_skill_bc --output outputs/v7_q3_skill_bc_<date>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256
from .train_vpt_heads import prefetch
from .v7_data import CACHE, SKILLS, FrameSampler, load_cache
from .vpt_features import load_vpt_policy
from .vpt_heads import SkillPolicy

VPT_MODEL = Path(__file__).resolve().parents[1] / 'checkpoints/foundation-model-3x.model'
VPT_WEIGHTS = Path(__file__).resolve().parents[1] / 'checkpoints/bc-early-game-3x.weights'


class SkillSampler(FrameSampler):
    """Half the batch from skill frames (balanced across skills), half uniform."""

    def __init__(self, files, seed):
        super().__init__(files, seed, event_boost=0.0)
        self.by_skill = {k: [] for k in range(1, len(SKILLS))}
        for i, f in enumerate(files):
            ok = ~f.null
            for k in self.by_skill:
                t = np.nonzero((f.skill == k) & ok)[0]
                t = t[t >= 1]
                if len(t):
                    self.by_skill[k].append((i, t))
        self.skills = [k for k, v in self.by_skill.items() if v]

    def sample(self, batch):
        idx = super().sample(batch // 2)
        for k in self.rng.choice(self.skills, batch - len(idx)):
            i, ts = self.by_skill[k][self.rng.integers(len(self.by_skill[k]))]
            idx.append((i, int(ts[self.rng.integers(len(ts))])))
        return idx


def joint_logprob(logp, buttons, camera):
    return logp['buttons'].gather(-1, buttons[:, None])[:, 0] + logp['camera'].gather(-1, camera[:, None])[:, 0]


@torch.no_grad()
def evaluate(policy, original, files, device, frames_per_skill=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(len(SKILLS)):
        pool = [(i, t) for i, f in enumerate(files) for t in np.nonzero((f.skill == k) & ~f.null)[0]]
        if not pool:
            continue
        pick = [pool[j] for j in rng.choice(len(pool), min(frames_per_skill, len(pool)), replace=False)]
        h = torch.from_numpy(np.stack([files[i].h[t] for i, t in pick]).astype(np.float32)).to(device)
        b = torch.tensor([files[i].buttons[t] for i, t in pick], dtype=torch.long, device=device)
        c = torch.tensor([files[i].camera[t] for i, t in pick], dtype=torch.long, device=device)
        s = torch.full((len(pick),), k, device=device)
        new = joint_logprob(policy(h, s), b, c).mean()
        old = joint_logprob(original(h, torch.zeros_like(s)), b, c).mean()
        out[SKILLS[k]] = dict(frames=len(pick), logprob_new=float(new), logprob_original=float(old), gain=float(new - old))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr-head', type=float, default=2e-5)
    parser.add_argument('--lr-film', type=float, default=1e-3)
    parser.add_argument('--beta-kl', type=float, default=0.1)
    parser.add_argument('--awr', type=Path, default=None, help='offline_rl weights.pt (Q1-OFF C0)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    vpt, *_ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL), weights_sha256=sha256(VPT_WEIGHTS), device='cpu')
    policy = SkillPolicy(vpt.pi_head).to(args.device)
    original = SkillPolicy(vpt.pi_head).to(args.device).requires_grad_(False)
    del vpt
    fit, select_files = load_cache({'fit'}), load_cache({'select'})
    awr = None
    if args.awr:
        awr = torch.load(args.awr, weights_only=False)
        weights = awr['weights']
        for f in fit:
            f.awr = weights.get(f.entry['stem'])
    config = dict(vars(args), output=str(args.output), awr=str(args.awr) if args.awr else None, cache_selection_sha256=sha256(CACHE / 'selection.jsonl'),
                  prior_weights_sha256=sha256(VPT_WEIGHTS), files=dict(fit=len(fit), select=len(select_files)),
                  code_sha256={p: sha256(Path(__file__).parent / p) for p in ('train_vpt_skill_bc.py', 'vpt_heads.py', 'v7_data.py')})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    sampler = SkillSampler(fit, args.seed)
    if awr is not None:
        base_gather = sampler.gather

        def gather(idx, device):
            batch = base_gather(idx, device)
            batch['weight'] = torch.tensor([fit[i].awr[t] if fit[i].awr is not None else 1.0 for i, t in idx], dtype=torch.float32)
            return batch
        sampler.gather = gather
    stream = prefetch(sampler, args.batch, args.device)
    opt = torch.optim.AdamW([dict(params=policy.pi_head.parameters(), lr=args.lr_head),
                             dict(params=list(policy.gamma.parameters()) + list(policy.beta.parameters()), lr=args.lr_film)],
                            weight_decay=0.0)
    log = (args.output / 'metrics.jsonl').open('x')
    started = time.time()
    for step in range(1, args.steps + 1):
        b = next(stream)
        valid = ~b['null']
        logp = policy(b['h'], b['skill'])
        nll = -joint_logprob(logp, b['buttons'], b['camera'])
        w = b['weight'] if 'weight' in b else torch.ones_like(nll)
        bc = (nll * w * valid).sum() / valid.sum().clamp(min=1)
        general = b['skill'] == 0
        kl = torch.zeros((), device=args.device)
        if general.any():
            ref = original(b['h'][general], b['skill'][general])
            cur = policy(b['h'][general], b['skill'][general])
            kl = sum((cur[k].exp() * (cur[k] - ref[k])).sum(-1).mean() for k in cur)
        loss = bc + args.beta_kl * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, bc=float(bc), kl_general=float(kl), sec=round(time.time() - started))) + '\n')
            log.flush()
        if step % 5000 == 0 or step == args.steps:
            torch.save(dict(pi_head=policy.pi_head.state_dict(), gamma=policy.gamma.state_dict(), beta=policy.beta.state_dict(),
                            skills=SKILLS, step=step, config=config), args.output / 'skill_policy.pt')
            print(step, round(float(bc), 4), round(float(kl), 5), flush=True)
    report = dict(config=config, select_eval=evaluate(policy, original, select_files, args.device))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['select_eval'], indent=1))


if __name__ == '__main__':
    main()
