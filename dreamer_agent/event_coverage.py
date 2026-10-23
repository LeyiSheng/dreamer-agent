"""Audit cumulative VPT counters without treating initial history as new events.

Task names follow Dreamer 4 Table 4 (arxiv:2509.24527). Events are inferred from
adjacent-frame stats deltas in the contractor jsonl; videos have no event IDs.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from .provenance import sha256

# Dreamer 4 Table 4 order. Each rule: (prefix, match) where match is an exact
# item suffix after "minecraft:", an endswith tuple, or a callable(key)->bool.
DREAMER4_TASKS = [
    "mine_log",
    "mine_cobblestone",
    "mine_iron_ore",
    "mine_coal",
    "mine_diamond",
    "craft_planks",
    "craft_stick",
    "craft_crafting_table",
    "craft_furnace",
    "craft_iron_ingot",
    "craft_wooden_pickaxe",
    "craft_stone_pickaxe",
    "craft_iron_pickaxe",
    "open_crafting_table",
    "open_furnace",
    "place_crafting_table",
    "place_furnace",
    "use_wooden_pickaxe",
    "use_stone_pickaxe",
    "use_iron_pickaxe",
]

def _endswith(suffixes):
    suffixes = tuple(suffixes)
    return lambda item: any(item.endswith(s) for s in suffixes)

def _exact(*names):
    names = set(names)
    return lambda item: item in names

# Maps task -> list of (stats_key_prefix, item_matcher).
# Prefer mine_block for "mine_*" to match Dreamer naming; pickup also accepted
# for drops that commonly appear without a matching mine_block key.
TASK_RULES = {
    "mine_log": [
        ("minecraft.mine_block:", _endswith(("_log",))),
        ("minecraft.pickup:", _endswith(("_log",))),
    ],
    "mine_cobblestone": [
        ("minecraft.pickup:", _exact("cobblestone")),
        ("minecraft.mine_block:", _exact("stone", "cobblestone")),
    ],
    "mine_iron_ore": [
        ("minecraft.mine_block:", _exact("iron_ore", "deepslate_iron_ore")),
        ("minecraft.pickup:", _exact("iron_ore", "deepslate_iron_ore", "raw_iron")),
    ],
    "mine_coal": [
        ("minecraft.mine_block:", _exact("coal_ore", "deepslate_coal_ore")),
        ("minecraft.pickup:", _exact("coal")),
    ],
    "mine_diamond": [
        ("minecraft.mine_block:", _exact("diamond_ore", "deepslate_diamond_ore")),
        ("minecraft.pickup:", _exact("diamond")),
    ],
    "craft_planks": [("minecraft.craft_item:", _endswith(("_planks",)))],
    "craft_stick": [("minecraft.craft_item:", _exact("stick"))],
    "craft_crafting_table": [("minecraft.craft_item:", _exact("crafting_table"))],
    "craft_furnace": [("minecraft.craft_item:", _exact("furnace"))],
    "craft_iron_ingot": [("minecraft.craft_item:", _exact("iron_ingot"))],
    "craft_wooden_pickaxe": [("minecraft.craft_item:", _exact("wooden_pickaxe"))],
    "craft_stone_pickaxe": [("minecraft.craft_item:", _exact("stone_pickaxe"))],
    "craft_iron_pickaxe": [("minecraft.craft_item:", _exact("iron_pickaxe"))],
    "open_crafting_table": [
        ("minecraft.custom:", _exact("interact_with_crafting_table")),
    ],
    "open_furnace": [("minecraft.custom:", _exact("interact_with_furnace"))],
    "place_crafting_table": [("minecraft.use_item:", _exact("crafting_table"))],
    "place_furnace": [("minecraft.use_item:", _exact("furnace"))],
    "use_wooden_pickaxe": [("minecraft.use_item:", _exact("wooden_pickaxe"))],
    "use_stone_pickaxe": [("minecraft.use_item:", _exact("stone_pickaxe"))],
    "use_iron_pickaxe": [("minecraft.use_item:", _exact("iron_pickaxe"))],
}

# Backward-compatible aliases used by earlier 5-task pipeline.
LEGACY_ALIASES = {
    "obtain_log": "mine_log",
    "obtain_cobblestone": "mine_cobblestone",
}

assert list(TASK_RULES) == DREAMER4_TASKS
TASKS = TASK_RULES  # exported name kept for older imports expecting TASKS dict


def task_for(key):
    if not isinstance(key, str) or ":" not in key:
        return None
    prefix, _, rest = key.partition(":")
    prefix = prefix + ":"
    item = rest.removeprefix("minecraft.")
    for task, rules in TASK_RULES.items():
        for rule_prefix, matcher in rules:
            if key.startswith(rule_prefix) and matcher(item):
                return task
    return None


def scan(rows):
    previous = None
    events = []
    quality = Counter()
    known = 0
    for i, row in enumerate(rows):
        stats = row.get("stats")
        valid = isinstance(stats, dict) and bool(stats) and all(
            type(v) is int and v >= 0 for v in stats.values())
        if not valid:
            quality["unavailable_stats_rows"] += 1
            previous = None
            continue
        known += 1
        if previous is None:
            quality["baseline_rows"] += 1
            previous = (i, row, stats)
            continue
        j, old, prev = previous
        if row.get("tick") != old.get("tick", -2) + 1 or row.get("serverTick", 0) < old.get("serverTick", 0):
            quality["time_discontinuities"] += 1
            previous = (i, row, stats)
            continue
        missing = set(prev) - set(stats)
        decreases = [k for k in prev.keys() & stats.keys()
                     if stats[k] < prev[k] and not k.startswith("minecraft.custom:")]
        if missing or decreases:
            quality["counter_discontinuities"] += 1
            previous = (i, row, stats)
            continue
        quality["comparable_transitions"] += 1
        for k, v in stats.items():
            task = task_for(k)
            if task is None:
                continue
            delta = v - prev.get(k, 0)
            if delta > 0:
                events.append(dict(
                    frame=i, previous_frame=j, tick=row.get("tick"),
                    server_tick=row.get("serverTick"), task=task, key=k, delta=delta,
                    kind=("existing_counter_increment" if k in prev else "new_key_candidate"),
                    uncertainty="event observed between adjacent recorded stats; not exact action causality"))
        previous = (i, row, stats)
    return dict(frames=len(rows), known_stats_rows=known, quality=dict(quality), events=events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    source = json.loads(a.sources.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for c in source["clips"]:
        if c["split"] not in ("train", "calibration"):
            raise ValueError("Test/evaluation data forbidden")
        p = Path(c["actions_path"])
        if sha256(p) != c["actions_sha256"]:
            raise ValueError("Raw actions changed")
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == c["stop"] and c["start"] == 0
        item = {k: c[k] for k in (
            "episode_id", "source_episode_id", "split", "actions_path",
            "actions_sha256", "video_path", "video_sha256")}
        item.update(scan(rows))
        records.append(item)
    summary = {}
    for split in ("train", "calibration"):
        ss = [r for r in records if r["split"] == split]
        summary[split] = {}
        for task in DREAMER4_TASKS:
            summary[split][task] = {
                kind: {
                    "events": sum(sum(e["task"] == task and e["kind"] == kind for e in r["events"]) for r in ss),
                    "sessions": len({r["source_episode_id"] for r in ss
                                     if any(e["task"] == task and e["kind"] == kind for e in r["events"])}),
                }
                for kind in ("existing_counter_increment", "new_key_candidate")}
    report = dict(
        schema_version=2, kind="offline_event_coverage_dreamer4_table4",
        sources_sha256=sha256(a.sources), code_sha256=sha256(__file__),
        tasks=DREAMER4_TASKS, task_rules={k: [(p, "matcher") for p, _ in v] for k, v in TASK_RULES.items()},
        records=records, summary=summary,
        limitations=[
            "Initial nonempty snapshot is baseline, not an event.",
            "New keys are candidates until sparse-counter semantics checked.",
            "No cross-clip differences; no exact physical event-action alignment claim.",
            "Task names follow Dreamer 4 Table 4; stats-key mapping is an engineering approximation of paper event annotations.",
            "No success claim for trained policy; stats are supervision only.",
        ])
    (a.output / "coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
