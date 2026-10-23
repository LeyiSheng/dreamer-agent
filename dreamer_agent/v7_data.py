"""Cached VPT-3x feature data for V7 heads, skill BC and offline RL (train split only).

Session groups (sha256 of source session, PLAN_V7 §8 Q1): digit 0-7 fit, 8 select
(thresholds, early stopping), 9 confirm (reported once). All groups are train-split
sessions; calibration/test sessions never enter the cache.
"""
import hashlib
import os
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('DREAMER_FEATURE_CACHE', str(ROOT / 'outputs/v7_vpt3x_feat_20260926'))).expanduser()
MILESTONES = ['log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe', 'cobblestone', 'stone_pickaxe',
              'furnace', 'iron_ore', 'iron_ingot', 'iron_pickaxe', 'diamond']
MILESTONE_TASK = {
    'mine_log': 'log', 'craft_planks': 'planks', 'craft_stick': 'stick', 'craft_crafting_table': 'crafting_table',
    'craft_wooden_pickaxe': 'wooden_pickaxe', 'mine_cobblestone': 'cobblestone', 'craft_stone_pickaxe': 'stone_pickaxe',
    'craft_furnace': 'furnace', 'mine_iron_ore': 'iron_ore', 'craft_iron_ingot': 'iron_ingot',
    'craft_iron_pickaxe': 'iron_pickaxe', 'mine_diamond': 'diamond',
}
# Inventory items in the cache (vpt_feature_cache.ITEMS order).
ITEMS = ['log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe', 'cobblestone', 'stone_pickaxe', 'furnace',
         'coal', 'iron_ore', 'iron_ingot', 'iron_pickaxe', 'diamond']
BUCKETS = (0, 1, 2, 3, 5, 9, 17)  # lower edges: 0 | 1 | 2 | 3-4 | 5-8 | 9-16 | 17+
SKILLS = ['general'] + MILESTONES
# Longest lookback (frames at 20 Hz) that counts as one skill segment before its completion event.
SKILL_MAX_FRAMES = {'log': 1200, 'planks': 400, 'stick': 400, 'crafting_table': 600, 'wooden_pickaxe': 800,
                    'cobblestone': 1200, 'stone_pickaxe': 800, 'furnace': 800, 'iron_ore': 2400, 'iron_ingot': 1200,
                    'iron_pickaxe': 800, 'diamond': 2400}


def session_group(session):
    digit = int(hashlib.sha256(session.encode()).hexdigest(), 16) % 10
    return 'fit' if digit < 8 else 'select' if digit == 8 else 'confirm'


def bucketize(counts):
    """int counts (-1 unknown) -> bucket index (-1 unknown)."""
    counts = np.asarray(counts)
    out = np.searchsorted(np.array(BUCKETS[1:]), counts, side='right')
    return np.where(counts < 0, -1, out)


def event_frames(events):
    """{milestone: sorted accepted event frames}."""
    out = {m: [] for m in MILESTONES}
    for e in events:
        m = MILESTONE_TASK.get(e['task'])
        if m:
            out[m].append(int(e['frame']))
    return {m: sorted(v) for m, v in out.items()}


def cluster_starts(frames, gap=200):
    starts, last = [], None
    for f in frames:
        if last is None or f - last > gap:
            starts.append(f)
        last = f
    return starts


def skill_labels(length, events):
    """Per-frame skill index: the frames leading to a milestone completion (bounded, not crossing an earlier completion)."""
    labels = np.zeros(length, np.int16)
    completions = sorted((f, m) for m, fs in event_frames(events).items() for f in cluster_starts(fs))
    previous = 0
    for frame, milestone in completions:
        start = max(previous, frame - SKILL_MAX_FRAMES[milestone])
        labels[start:frame + 1] = SKILLS.index(milestone)
        previous = frame + 1
    return labels


class CachedFile:
    def __init__(self, entry):
        self.entry = entry
        base = CACHE / 'files' / entry['stem']
        self.h = np.load(f'{base}.h.npy', mmap_mode='r')
        with np.load(f'{base}.npz') as z:
            self.buttons, self.camera = z['buttons'], z['camera']
            self.null, self.gui = z['null'], z['gui']
            self.inventory = z['inventory']
            self.events = json.loads(str(z['events']))
        self.length = len(self.h)
        self.group = session_group(entry['session'])
        ev = event_frames(self.events)
        self.event = np.zeros((self.length, len(MILESTONES)), np.uint8)
        for k, m in enumerate(MILESTONES):
            for f in ev[m]:
                if 0 <= f < self.length:
                    self.event[max(0, f - 2):f + 3, k] = 1
        self.event_frames = ev
        self.belief = bucketize(self.inventory).astype(np.int8)
        self.skill = skill_labels(self.length, self.events)


def load_cache(groups=None, limit=None):
    entries = [json.loads(l) for l in (CACHE / 'selection.jsonl').read_text().splitlines()]
    files = []
    for e in entries:
        if not (CACHE / 'files' / f"{e['stem']}.npz").exists():
            continue
        if groups and session_group(e['session']) not in groups:
            continue
        files.append(CachedFile(e))
        if limit and len(files) >= limit:
            break
    return files


class FrameSampler:
    """Uniform random frames (t >= 1) across files, optionally weighted toward positive events."""

    def __init__(self, files, seed, event_boost=0.3):
        self.files = files
        self.rng = np.random.default_rng(seed)
        self.lengths = np.array([f.length for f in files])
        self.cum = np.cumsum(self.lengths)
        self.positives = [(i, t) for i, f in enumerate(files) for t in np.nonzero(f.event.any(1))[0] if t >= 1]
        self.event_boost = event_boost

    def sample(self, batch):
        n_pos = int(batch * self.event_boost) if self.positives else 0
        idx = []
        for j in self.rng.integers(0, len(self.positives), n_pos):
            idx.append(self.positives[j])
        for g in self.rng.integers(0, self.cum[-1], batch - n_pos):
            i = int(np.searchsorted(self.cum, g, side='right'))
            t = int(g - (self.cum[i - 1] if i else 0))
            idx.append((i, max(1, t)))
        return idx

    def gather(self, idx, device):
        ii = np.fromiter((i for i, _ in idx), np.int64, len(idx))
        tt = np.fromiter((t for _, t in idx), np.int64, len(idx))
        n, dim = len(idx), self.files[0].h.shape[1]
        fields = dict(h=np.empty((n, dim), np.float32), h_prev=np.empty((n, dim), np.float32),
                      event=np.empty((n, len(MILESTONES)), np.float32), belief=np.empty((n, len(ITEMS)), np.int64),
                      buttons=np.empty(n, np.int64), camera=np.empty(n, np.int64), null=np.empty(n, bool), skill=np.empty(n, np.int64))
        order = np.argsort(ii, kind='stable')
        bounds = np.flatnonzero(np.diff(ii[order])) + 1
        for rows in np.split(order, bounds):
            f, t = self.files[ii[rows[0]]], tt[rows]
            both = f.h[np.concatenate([t, t - 1])]
            fields['h'][rows], fields['h_prev'][rows] = both[:len(t)], both[len(t):]
            fields['event'][rows], fields['belief'][rows] = f.event[t], f.belief[t]
            fields['buttons'][rows], fields['camera'][rows] = f.buttons[t], f.camera[t]
            fields['null'][rows], fields['skill'][rows] = f.null[t], f.skill[t]
        return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in fields.items()}
