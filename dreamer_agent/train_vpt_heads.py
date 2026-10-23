"""V7 Q1: train event/belief ensemble on cached real features, calibrate the complete reward
rule on 'select' sessions, confirm once on 'confirm' sessions (PLAN_V7 §8 Q1).

  python -m dreamer_agent.train_vpt_heads --output outputs/v7_q1_heads_<date> [--steps N]
"""
import argparse
import itertools
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .provenance import sha256
from .reward_rule import RuleParams, detect, match, wilson
from .v7_data import CACHE, ITEMS, MILESTONES, FrameSampler, cluster_starts, load_cache
from .vpt_heads import SignalHeads

PRECISION_TARGET = 0.9       # threshold selection on 'select'
CONFIRM_LOWER_BOUND = 0.8    # gate: Wilson 95% lower bound of precision on 'confirm'
MIN_CLUSTERS = 20            # minimum confirm support (event clusters) to be usable as RL reward
GRID_E = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
GRID_C = (0.3, 0.5, 0.7, 0.9)


class _Stream(torch.utils.data.IterableDataset):
    def __init__(self, sampler, batch):
        self.sampler, self.batch = sampler, batch
        self.seed = int(sampler.rng.integers(2 ** 31))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        self.sampler.rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield self.sampler.gather(self.sampler.sample(self.batch), device='cpu')


def prefetch(sampler, batch, device, depth=4, workers=4):
    """Worker processes (not threads): gathering from many mmaps is GIL-bound."""
    loader = torch.utils.data.DataLoader(_Stream(sampler, batch), batch_size=None, num_workers=workers,
                                         prefetch_factor=depth, pin_memory=True, persistent_workers=True)
    for b in loader:
        yield {k: v.to(device, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def file_signals(heads, f, device, chunk=4096):
    out = {k: [] for k in ('event_mean', 'event_std', 'latch_mean', 'latch_std', 'belief')}
    h_all = torch.from_numpy(np.asarray(f.h, dtype=np.float32))
    prev = torch.cat([h_all[:1], h_all[:-1]])
    for s in range(0, len(h_all), chunk):
        sig = heads.signals(h_all[s:s + chunk].to(device), prev[s:s + chunk].to(device))
        for k in out:
            out[k].append(sig[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_rule(signals_by_file, files, params, milestone):
    tp = n_det = hit = n_clusters = 0
    sessions = set()
    for f, sig in zip(files, signals_by_file):
        clusters = cluster_starts(f.event_frames[milestone])
        a, b, c, d = match(detect(sig, params, milestone), clusters, params.confirm_frames)
        tp, n_det, hit, n_clusters = tp + a, n_det + b, hit + c, n_clusters + d
        if clusters:
            sessions.add(f.entry['session'])
    return dict(tp=tp, detections=n_det, clusters_hit=hit, clusters=n_clusters, sessions=len(sessions),
                precision=tp / n_det if n_det else None, recall=hit / n_clusters if n_clusters else None,
                precision_wilson95=wilson(tp, n_det))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    fit = load_cache({'fit'})
    select_files = load_cache({'select'})
    confirm_files = load_cache({'confirm'})
    config = dict(vars(args), output=str(args.output), cache=str(CACHE), cache_selection_sha256=sha256(CACHE / 'selection.jsonl'),
                  code_sha256={p: sha256(Path(__file__).parent / p) for p in ('train_vpt_heads.py', 'vpt_heads.py', 'reward_rule.py', 'v7_data.py')},
                  files=dict(fit=len(fit), select=len(select_files), confirm=len(confirm_files)),
                  precision_target=PRECISION_TARGET, confirm_lower_bound=CONFIRM_LOWER_BOUND, min_clusters=MIN_CLUSTERS)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config['files']), flush=True)
    heads = SignalHeads(args.members).to(args.device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    samplers = [FrameSampler(fit, seed=args.seed * 100 + m) for m in range(args.members)]
    streams = [prefetch(s, args.batch, args.device) for s in samplers]
    pos_rate = np.mean([f.event.mean(0) for f in fit], 0)
    log = (args.output / 'metrics.jsonl').open('x')
    started = time.time()
    for step in range(1, args.steps + 1):
        loss_total, parts = 0, {}
        for m, stream in enumerate(streams):  # bootstrap: each member sees its own sample stream
            b = next(stream)
            event, belief = heads.members[m](b['h'], b['h_prev'])
            ev_loss = F.binary_cross_entropy_with_logits(event, b['event'], pos_weight=torch.full_like(event[0], 5.0))
            mask = b['belief'] >= 0
            bl = F.cross_entropy(belief[mask], b['belief'][mask]) if mask.any() else event.sum() * 0
            loss_total = loss_total + ev_loss + bl
            parts[f'event{m}'], parts[f'belief{m}'] = float(ev_loss), float(bl)
        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, loss=float(loss_total), **parts, sec=round(time.time() - started))) + '\n')
            log.flush()
        if step % 2000 == 0:
            print(step, round(float(loss_total), 4), round(time.time() - started), flush=True)
    torch.save(dict(state=heads.state_dict(), members=args.members, config=config), args.output / 'heads.pt')

    heads.eval()
    sel = [file_signals(heads, f, args.device) for f in select_files]
    params = RuleParams()
    selection = {}
    for m in MILESTONES:
        best = None
        for te, tc in itertools.product(GRID_E, GRID_C):
            params.event_threshold[m], params.latch_threshold[m] = te, tc
            r = evaluate_rule(sel, select_files, params, m)
            ok = r['precision'] is not None and r['precision'] >= PRECISION_TARGET
            key = (ok, r['recall'] or 0, te)
            if best is None or key > best[0]:
                best = (key, te, tc, r)
        _, params.event_threshold[m], params.latch_threshold[m], selection[m] = best
    del sel
    conf = [file_signals(heads, f, args.device) for f in confirm_files]
    confirm = {m: evaluate_rule(conf, confirm_files, params, m) for m in MILESTONES}
    # Belief accuracy on confirm (frame-level, known inventory only).
    belief_acc = {}
    for i, item in enumerate(ITEMS):
        correct = total = 0
        for f, sig in zip(confirm_files, conf):
            known = f.belief[:, i] >= 0
            correct += int((sig['belief'][known, i].argmax(-1) == f.belief[known, i]).sum())
            total += int(known.sum())
        belief_acc[item] = correct / total if total else None
    usable = {m: bool(confirm[m]['precision_wilson95'][0] is not None and confirm[m]['precision_wilson95'][0] >= CONFIRM_LOWER_BOUND
                      and confirm[m]['clusters'] >= MIN_CLUSTERS) for m in MILESTONES}
    report = dict(config=config, rule=params.to_dict(), selection_on_select=selection, confirm=confirm,
                  belief_argmax_accuracy_confirm=belief_acc, usable_for_rl_reward=usable,
                  gate_early=dict((m, usable[m]) for m in ('log', 'crafting_table', 'wooden_pickaxe', 'cobblestone', 'stone_pickaxe')),
                  note='Real-frame calibration only (Q1). Generated-domain precision requires Q2 human annotation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    (args.output / 'rule.json').write_text(json.dumps(params.to_dict(), indent=2))
    print(json.dumps(dict(usable=usable, confirm={m: (c['precision'], c['recall'], c['clusters']) for m, c in confirm.items()}), indent=1))


if __name__ == '__main__':
    main()
