"""V7 Q1 train-only milestone support audit (PLAN_V7 §4.1, §4.2, §8 Q1).

Reads the per-file accepted stats events of the p1c full-corpus audit. Only
status=eligible train files are counted; calibration/test files are skipped
without reading their events. Support is reported in source sessions, files and
event clusters, never in frames, because adjacent frames are not independent.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'outputs/p1c_dreamer4_table4_audit_v1/training_records.jsonl'
CLUSTER_GAP = 200  # frames (10 s) between accepted events of one task that start a new cluster
MILESTONES = {
    'log': 'mine_log', 'planks': 'craft_planks', 'stick': 'craft_stick', 'crafting_table': 'craft_crafting_table',
    'wooden_pickaxe': 'craft_wooden_pickaxe', 'cobblestone': 'mine_cobblestone', 'stone_pickaxe': 'craft_stone_pickaxe',
    'furnace': 'craft_furnace', 'iron_ore': 'mine_iron_ore', 'iron_ingot': 'craft_iron_ingot',
    'iron_pickaxe': 'craft_iron_pickaxe', 'diamond': 'mine_diamond',
}
ORDER = list(MILESTONES)
EXTRA_TASKS = ('mine_coal', 'open_crafting_table', 'open_furnace', 'place_crafting_table', 'place_furnace',
               'use_wooden_pickaxe', 'use_stone_pickaxe', 'use_iron_pickaxe')


def clusters(frames):
    count, last = 0, None
    for frame in sorted(frames):
        if last is None or frame - last > CLUSTER_GAP:
            count += 1
        last = frame
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    splits, statuses, versions = Counter(), Counter(), Counter()
    tasks = [*MILESTONES.values(), *EXTRA_TASKS]
    support = {t: dict(files=0, sessions=set(), clusters=0, events=0) for t in tasks}
    diamond_keys = Counter()
    first_frames = defaultdict(list)
    chain_depth = Counter()
    train_frames, train_files, train_sessions = 0, 0, set()
    rare = []
    with RECORDS.open() as stream:
        for line in stream:
            record = json.loads(line)
            splits[record['split']] += 1
            statuses[record['status']] += 1
            if record['split'] != 'train' or record['status'] != 'eligible_pending_full_decode':
                continue
            train_files += 1
            train_frames += record['frames']
            train_sessions.add(record['source_session'])
            versions[record['version']] += 1
            by_task = defaultdict(list)
            for event in record['events']:
                if event.get('accepted'):
                    by_task[event['task']].append(event['frame'])
                    if event['task'] == 'mine_diamond':
                        diamond_keys[event['key']] += 1
            for task in tasks:
                if by_task[task]:
                    entry = support[task]
                    entry['files'] += 1
                    entry['sessions'].add(record['source_session'])
                    entry['clusters'] += clusters(by_task[task])
                    entry['events'] += len(by_task[task])
            reached = {m: min(by_task[t]) for m, t in MILESTONES.items() if by_task[t]}
            for m, frame in reached.items():
                first_frames[m].append(frame)
            depth = 0
            for m in ORDER:
                if m in reached and (depth == 0 or reached[m] >= reached[ORDER[depth - 1]]):
                    depth += 1
                else:
                    break
            chain_depth[depth] += 1
            if 'iron_pickaxe' in reached or 'diamond' in reached:
                rare.append(dict(stem=record['stem'], version=record['version'], session=record['source_session'],
                                 frames=record['frames'], reached_first_frame=reached))

    def pct(values, q):
        return float(sorted(values)[int(q * (len(values) - 1))]) if values else None

    report = dict(
        schema_version=1, scope='v7_q1_train_support', records=str(RECORDS), records_sha256=sha256(RECORDS),
        code_sha256=sha256(__file__), cluster_gap_frames=CLUSTER_GAP,
        split_counts_all=dict(splits), status_counts_all=dict(statuses),
        train=dict(files=train_files, sessions=len(train_sessions), frames=train_frames, hours=train_frames / 20 / 3600,
                   versions=dict(versions)),
        milestone_support={m: dict(task=t, files=support[t]['files'], sessions=len(support[t]['sessions']),
                                   clusters=support[t]['clusters'], events=support[t]['events'],
                                   first_frame_in_file_p10_p50_p90=[pct(first_frames[m], q) for q in (.1, .5, .9)])
                           for m, t in MILESTONES.items()},
        extra_task_support={t: dict(files=support[t]['files'], sessions=len(support[t]['sessions']),
                                    clusters=support[t]['clusters'], events=support[t]['events']) for t in EXTRA_TASKS},
        diamond_event_keys=dict(diamond_keys),
        ordered_prefix_chain_depth_per_file={ORDER[d - 1] if d else 'none': chain_depth[d] for d in range(len(ORDER) + 1)},
        limitations=('Events are stats deltas accepted by the p1c audit rules (inventory corroboration within 20 frames for '
                     'new keys); files are recording segments, so first_frame is from segment start, not from world spawn; '
                     'a file-level chain depth undercounts chains that span segments. Held-out/calibration events not read.'))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'rare_files.jsonl').open('x') as f:
        for row in rare:
            f.write(json.dumps(row) + '\n')
    print(json.dumps({k: report[k] for k in ('train', 'milestone_support', 'ordered_prefix_chain_depth_per_file', 'diamond_event_keys')}, indent=1))


if __name__ == '__main__':
    main()
