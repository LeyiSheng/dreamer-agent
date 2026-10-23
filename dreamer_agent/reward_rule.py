"""V7 complete reward rule (PLAN_V7 §4.2-§4.4): ensemble -> threshold -> event candidate
-> latch confirmation within m frames -> refractory/ledger dedupe. Calibration targets
the rule's final output, not raw sigmoids.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .diamond_metric import debounced_high_water
from .v7_data import ITEMS, MILESTONES

MILESTONE_ITEM = {m: ITEMS.index(m) for m in MILESTONES}
REFRACTORY = 200  # frames; one reward per event cluster (matches the support-count cluster gap)


@dataclass
class RuleParams:
    event_threshold: dict = field(default_factory=lambda: {m: 0.5 for m in MILESTONES})
    latch_threshold: dict = field(default_factory=lambda: {m: 0.5 for m in MILESTONES})
    count_delta: dict = field(default_factory=lambda: {m: 0.0 for m in MILESTONES})
    kappa: float = 0.0
    confirm_frames: int = 20
    count_window: int = 60
    diamond_hold: int = 10

    def to_dict(self):
        return dict(event_threshold=self.event_threshold, latch_threshold=self.latch_threshold, count_delta=self.count_delta,
                    kappa=self.kappa, confirm_frames=self.confirm_frames, count_window=self.count_window, diamond_hold=self.diamond_hold)


def detect(signals, params, milestone):
    """signals: dict of numpy arrays over time ([T,K] event_mean/std, [T,I] latch_mean, [T,I] count_lower).
    A candidate is confirmed at the first frame within confirm_frames where the latch exceeds theta_c
    and the believed count lower bound rose by >= count_delta since count_window frames before the
    candidate. Returns reward frames."""
    k, item = MILESTONES.index(milestone), MILESTONE_ITEM[milestone]
    score = signals['event_mean'][:, k] - params.kappa * signals['event_std'][:, k]
    latch = signals['latch_mean'][:, item]
    theta_e, theta_c = params.event_threshold[milestone], params.latch_threshold[milestone]
    delta = params.count_delta.get(milestone, 0.0)
    count = signals['count_lower'][:, item] if delta > 0 else None
    out, last, t, length = [], -10 ** 9, 0, len(score)
    candidates = np.nonzero(score > theta_e)[0]
    for c in candidates:
        if c - last < REFRACTORY:
            continue
        end = min(length, c + params.confirm_frames + 1)
        ok = latch[c:end] > theta_c
        if count is not None:
            ok &= count[c:end] - count[max(0, c - params.count_window)] >= delta
        hit = np.nonzero(ok)[0]
        if len(hit):
            frame = int(c + hit[0])
            out.append(frame)
            last = frame
    return out


def diamond_rewards(signals, params):
    """Increment of the debounced high-water of the believed diamond count (argmax bucket lower edge)."""
    from .v7_data import BUCKETS
    belief = signals['belief'][:, ITEMS.index('diamond')]
    counts = [int(BUCKETS[int(np.argmax(b))]) if b.max() > 0.5 else None for b in belief]
    return debounced_high_water(counts, params.diamond_hold)


def match(detections, event_frames, confirm_frames, slack=20, gap=200):
    """A detection is TP if it lies in [f - slack, f + confirm + slack] of ANY true event frame f
    (repeated acquisitions inside one cluster are real events, not false positives). Recall counts
    clusters (events with gaps <= gap) hit by at least one detection."""
    ev = np.asarray(sorted(event_frames), dtype=np.int64)
    det = np.asarray(detections, dtype=np.int64)
    tp = 0
    if len(ev) and len(det):
        lo = np.searchsorted(ev, det - confirm_frames - slack, side='left')
        hi = np.searchsorted(ev, det + slack, side='right')
        tp = int((hi > lo).sum())
    clusters, hit = [], 0
    for f in ev:
        if clusters and f - clusters[-1][1] <= gap:
            clusters[-1][1] = f
        else:
            clusters.append([f, f])
    for start, end in clusters:
        if len(det) and ((det >= start - slack) & (det <= end + confirm_frames + slack)).any():
            hit += 1
    return tp, len(det), hit, len(clusters)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]
