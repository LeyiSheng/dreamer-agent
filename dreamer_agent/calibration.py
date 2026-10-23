"""P1 reward threshold calibration with precision uncertainty and counts."""
import math
import numpy as np


def wilson(successes, total, z=1.959963984540054):
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    if not total:
        return None
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def metrics(scores, labels, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or scores.size == 0 or labels.shape != scores.shape:
        raise ValueError("Expected nonempty score/label vectors")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be probabilities, labels binary")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    predicted, actual = scores >= threshold, labels.astype(bool)
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                precision_interval=wilson(tp, tp+fp), brier=float(np.mean((scores-labels)**2)),
                n=len(scores))


def select_threshold(scores, labels, *, precision_lower=.95, min_predictions=30):
    """Select on calibration only. Null threshold means no evidence to enable reward."""
    if not 0 < precision_lower < 1 or min_predictions < 1:
        raise ValueError("Invalid calibration requirements")
    metrics(scores, labels, .5)
    # One sort and cumulative counts, instead of rescanning N examples N times.
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(-scores, kind="stable")
    ranked, truth = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked) != 0), len(ranked)-1]
    tp = np.cumsum(truth)[ends]
    n = ends + 1
    precision = tp / n
    z = 1.959963984540054
    lower = (precision + z*z/(2*n) - z*np.sqrt(precision*(1-precision)/n + z*z/(4*n*n))) / (1+z*z/n)
    eligible = np.flatnonzero((n >= min_predictions) & (lower >= precision_lower))
    if not len(eligible):
        return dict(threshold=None, passed=False, reason="Insufficient precision evidence")
    total_positive = max(int(labels.sum()), 1)
    best_index = max(eligible, key=lambda i: (tp[i]/total_positive, lower[i]))
    best = metrics(scores, labels, float(ranked[ends[best_index]]))
    return dict(best, passed=True, scope="threshold selection only; IID Wilson assumption; requires independent confirmation and episode correlation analysis")
