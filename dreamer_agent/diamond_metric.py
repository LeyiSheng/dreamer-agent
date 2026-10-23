"""Diamond scoring definitions frozen in Q0 (PLAN_V7 §4.4).

Evaluation: high-water mark of the true inventory diamond count within an episode.
Training: high-water mark of a debounced belief count; the reward is its increment.
Both measure the highest simultaneous holding, not cumulative mining: consumption,
drops and death never lower the mark, and re-acquiring diamonds only pays once the
previous peak is exceeded.
"""


def high_water(counts):
    """Running maximum of per-step true counts (evaluation metric)."""
    mark, out = 0, []
    for count in counts:
        mark = max(mark, int(count))
        out.append(mark)
    return out


def debounced_high_water(beliefs, hold):
    """Running high-water of a belief count that must hold for `hold` consecutive steps.

    beliefs: per-step integer count or None (unknown). A level c is confirmed at t when
    every belief in [t-hold+1, t] is known and >= c. Unknown steps confirm nothing.
    Returns (marks, rewards) with rewards[t] = marks[t] - marks[t-1].
    """
    if hold < 1:
        raise ValueError('hold must be >= 1')
    mark, marks, rewards, window = 0, [], [], []
    for belief in beliefs:
        window = (window + [belief])[-hold:]
        before = mark
        if len(window) == hold and all(b is not None for b in window):
            mark = max(mark, min(int(b) for b in window))
        marks.append(mark)
        rewards.append(mark - before)
    return marks, rewards
