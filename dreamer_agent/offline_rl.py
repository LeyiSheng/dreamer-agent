"""V7 Q1-OFF: advantage weights for real-data offline RL control C0 (PLAN_V7 §8 Q1-OFF).

Real stats events are ground truth on real video. For every skill-labelled frame t of a
segment completed at frame f, the local return is gamma_low^(f - t) (first completion,
cont = 0 afterwards). A twohot value V(h, skill, progress) is fitted on 'fit' sessions;
AWR weights w = clip(exp((R - V) / beta), 0, w_max) are written per file and consumed
by train_vpt_skill_bc --awr. General frames keep weight 1.

Limitation: segments are defined backwards from completions, so there are no failed
attempts; the advantage ranks speed-to-completion, not success vs failure.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256
from .v7_data import CACHE, SKILL_MAX_FRAMES, SKILLS, load_cache
from .vpt_heads import TwoHotValue, value_input


def segment_targets(f, gamma):
    """Per-frame (return, progress) for skill frames; NaN elsewhere."""
    ret = np.full(f.length, np.nan, np.float32)
    prog = np.zeros(f.length, np.float32)
    t = 0
    while t < f.length:
        k = int(f.skill[t])
        if k == 0:
            t += 1
            continue
        end = t
        while end + 1 < f.length and f.skill[end + 1] == k:
            end += 1
        steps = np.arange(end - t, -1, -1)
        ret[t:end + 1] = gamma ** steps
        prog[t:end + 1] = np.arange(end - t + 1) / SKILL_MAX_FRAMES[SKILLS[k]]
        t = end + 1
    return ret, np.minimum(prog, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gamma', type=float, default=0.9995)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--w-max', type=float, default=20.0)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = load_cache({'fit'})
    targets = [segment_targets(f, args.gamma) for f in files]
    pool = [(i, t) for i, (r, _) in enumerate(targets) for t in np.nonzero(~np.isnan(r))[0]]
    rng = np.random.default_rng(0)
    value = TwoHotValue().to(args.device)
    opt = torch.optim.AdamW(value.parameters(), lr=3e-4)
    started = time.time()

    def batch(idx):
        h = torch.from_numpy(np.stack([files[i].h[t] for i, t in idx]).astype(np.float32)).to(args.device)
        k = torch.tensor([int(files[i].skill[t]) for i, t in idx], device=args.device)
        p = torch.tensor([targets[i][1][t] for i, t in idx], device=args.device)
        r = torch.tensor([targets[i][0][t] for i, t in idx], device=args.device)
        return value_input(h, k, p, torch.ones_like(p)), r

    for step in range(1, args.steps + 1):
        x, r = batch([pool[j] for j in rng.integers(0, len(pool), args.batch)])
        loss = value.loss(x, r).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print(step, float(loss), round(time.time() - started), flush=True)
    weights, stats = {}, []
    with torch.no_grad():
        for f, (r, p) in zip(files, targets):
            w = np.ones(f.length, np.float32)
            idx = np.nonzero(~np.isnan(r))[0]
            for s in range(0, len(idx), 8192):
                chunk = idx[s:s + 8192]
                h = torch.from_numpy(np.asarray(f.h[chunk], dtype=np.float32)).to(args.device)
                k = torch.from_numpy(f.skill[chunk].astype(np.int64)).to(args.device)
                pp = torch.from_numpy(p[chunk]).to(args.device)
                v = value.value(value_input(h, k, pp, torch.ones_like(pp))).cpu().numpy()
                adv = r[chunk] - v
                w[chunk] = np.clip(np.exp(adv / args.beta), 0, args.w_max)
                stats.append(adv)
            weights[f.entry['stem']] = w
    adv = np.concatenate(stats) if stats else np.zeros(1)
    torch.save(dict(weights=weights, value=value.state_dict(), config=vars(args) | dict(output=str(args.output))), args.output / 'weights.pt')
    report = dict(files=len(files), skill_frames=len(pool), cache_selection_sha256=sha256(CACHE / 'selection.jsonl'),
                  code_sha256=sha256(__file__), advantage_mean=float(adv.mean()), advantage_std=float(adv.std()),
                  weight_mean_skill=float(np.mean(np.clip(np.exp(adv / args.beta), 0, args.w_max))), beta=args.beta, gamma=args.gamma)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
