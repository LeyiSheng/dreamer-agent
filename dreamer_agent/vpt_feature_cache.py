"""V7 frozen VPT-3x feature cache on a pre-registered train-only subset (PLAN_V7 §3.1, Q1/Q1-OFF/Q3).

Every frame of a selected file passes through the backbone in order (as at deployment);
null actions are kept in the stream and only masked in BC losses. Frame preparation
follows the official VPT data loader: stuck-attack removal, hotbar-change actions,
cursor overlay when a GUI is open, BGR->RGB, INTER_LINEAR 128x128.

  python -m dreamer_agent.vpt_feature_cache select --output DIR --budget-frames N
  torchrun ... -m dreamer_agent.vpt_feature_cache extract --output DIR   (one rank per GPU)

Per file: {stem}.h.npy float16 [T,3072]; {stem}.npz with buttons/camera int16 [T],
null/gui bool [T], inventory int16 [T,len(ITEMS)] (-1 = unavailable), accepted events.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'outputs/p1c_dreamer4_table4_audit_v1/training_records.jsonl'
VPT_ROOT = ROOT / 'third_party/Video-Pre-Training'
VPT_MODEL = ROOT / 'checkpoints/foundation-model-3x.model'
VPT_WEIGHTS = ROOT / 'checkpoints/bc-early-game-3x.weights'
CHUNK = 256
WOODS = ('oak', 'birch', 'spruce', 'jungle', 'acacia', 'dark_oak')
ITEM_GROUPS = {
    'log': tuple(f'{w}_log' for w in WOODS) + tuple(f'stripped_{w}_log' for w in WOODS),
    'planks': tuple(f'{w}_planks' for w in WOODS + ('crimson', 'warped')),
    'stick': ('stick',), 'crafting_table': ('crafting_table',), 'wooden_pickaxe': ('wooden_pickaxe',),
    'cobblestone': ('cobblestone',), 'stone_pickaxe': ('stone_pickaxe',), 'furnace': ('furnace',),
    'coal': ('coal', 'charcoal'), 'iron_ore': ('iron_ore',), 'iron_ingot': ('iron_ingot',),
    'iron_pickaxe': ('iron_pickaxe',), 'diamond': ('diamond',),
}
ITEMS = list(ITEM_GROUPS)
ADVANCED = ('craft_crafting_table', 'craft_wooden_pickaxe', 'craft_stone_pickaxe', 'craft_furnace', 'mine_iron_ore',
            'craft_iron_ingot', 'craft_iron_pickaxe', 'mine_diamond')
EARLY = ('mine_log', 'craft_planks', 'craft_stick', 'mine_cobblestone')

# Official VPT json -> MineRL action conversion (run_inverse_dynamics_model.py), copied to
# avoid importing its MineRL-dependent module.
KEYBOARD_BUTTON_MAPPING = {
    'key.keyboard.escape': 'ESC', 'key.keyboard.s': 'back', 'key.keyboard.q': 'drop', 'key.keyboard.w': 'forward',
    **{f'key.keyboard.{i}': f'hotbar.{i}' for i in range(1, 10)},
    'key.keyboard.e': 'inventory', 'key.keyboard.space': 'jump', 'key.keyboard.a': 'left', 'key.keyboard.d': 'right',
    'key.keyboard.left.shift': 'sneak', 'key.keyboard.left.control': 'sprint', 'key.keyboard.f': 'swapHands',
}
NOOP = {**{v: 0 for v in KEYBOARD_BUTTON_MAPPING.values()}, 'attack': 0, 'use': 0, 'pickItem': 0}
CAMERA_SCALER = 360.0 / 2400.0
MINEREC_ORIGINAL_HEIGHT_PX = 720


def json_action_to_env_action(row):
    action = dict(NOOP, camera=np.array([0.0, 0.0]))
    null = True
    for key in row['keyboard']['keys']:
        if key in KEYBOARD_BUTTON_MAPPING:
            action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            null = False
    mouse = row['mouse']
    action['camera'] = np.array([mouse['dy'] * CAMERA_SCALER, mouse['dx'] * CAMERA_SCALER])
    if mouse['dx'] != 0 or mouse['dy'] != 0:
        null = False
    elif abs(action['camera'][0]) > 180 or abs(action['camera'][1]) > 180:
        action['camera'] = np.array([0.0, 0.0])
    for button, name in ((0, 'attack'), (1, 'use'), (2, 'pickItem')):
        if button in mouse['buttons']:
            action[name] = 1
            null = False
    return action, null


def official_actions(rows):
    """Per-row MineRL env actions with the official stuck-attack and hotbar fixes."""
    out, stuck, last_hotbar = [], False, 0
    for i, row in enumerate(rows):
        row = dict(row, mouse=dict(row['mouse']))
        if i == 0:
            stuck = row['mouse']['newButtons'] == [0]
        elif stuck and 0 in row['mouse']['newButtons']:
            stuck = False
        if stuck:
            row['mouse']['buttons'] = [b for b in row['mouse']['buttons'] if b != 0]
        action, null = json_action_to_env_action(row)
        hotbar = row.get('hotbar', last_hotbar)
        if hotbar != last_hotbar:
            action[f'hotbar.{hotbar + 1}'] = 1
        last_hotbar = hotbar
        out.append((action, null))
    return out


def inventory_row(row):
    inventory = row.get('inventory')
    if not isinstance(inventory, list):
        return [-1] * len(ITEMS)
    counts = {}
    for entry in inventory:
        if isinstance(entry, dict) and isinstance(entry.get('type'), str) and type(entry.get('quantity')) is int:
            name = entry['type'].removeprefix('minecraft:')
            counts[name] = counts.get(name, 0) + entry['quantity']
    return [sum(counts.get(i, 0) for i in ITEM_GROUPS[g]) for g in ITEMS]


def stable_hash(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------- selection

def select(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tier_a, tier_b = [], []
    with RECORDS.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['split'] != 'train' or r['status'] != 'eligible_pending_full_decode':
                continue
            tasks = {e['task'] for e in r['events'] if e.get('accepted')}
            entry = dict(stem=r['stem'], version=r['version'], session=r['source_session'], video=r['video_path'],
                         actions=r['actions_path'], actions_sha256=r['actions_sha256'], frames=r['frames'],
                         tasks=sorted(tasks), events=[dict(frame=e['frame'], task=e['task'], key=e['key'], delta=e['delta'])
                                                      for e in r['events'] if e.get('accepted')])
            (tier_a if tasks & set(ADVANCED) else tier_b).append(entry)
    # Round-robin over advanced milestones so every one gets files, not only the long iron/diamond ones.
    queues = {t: sorted((e for e in tier_a if t in e['tasks']), key=lambda e: stable_hash(t + e['stem'])) for t in ADVANCED}
    chosen, seen, total = [], set(), 0
    cap = args.budget_frames - args.background_frames
    while any(queues.values()) and total < cap:
        for task in ADVANCED:
            while queues[task]:
                e = queues[task].pop(0)
                if e['stem'] in seen:
                    continue
                if total + e['frames'] <= cap:
                    chosen.append(dict(e, tier='A'))
                    seen.add(e['stem'])
                    total += e['frames']
                break
    for e in sorted(tier_b, key=lambda e: stable_hash('b:' + e['stem'])):
        if total + e['frames'] <= args.budget_frames:
            chosen.append(dict(e, tier='B'))
            total += e['frames']
    with (args.output / 'selection.jsonl').open('x') as f:
        for e in chosen:
            f.write(json.dumps(e) + '\n')
    summary = dict(schema_version=1, records=str(RECORDS), records_sha256=sha256(RECORDS), code_sha256=sha256(__file__),
                   budget_frames=args.budget_frames, background_frames=args.background_frames, files=len(chosen), frames=total,
                   tier_a_files_available=len(tier_a), tier_a_frames_available=sum(e['frames'] for e in tier_a),
                   tier_a_files=sum(e['tier'] == 'A' for e in chosen), tier_b_files=sum(e['tier'] == 'B' for e in chosen),
                   sessions=len({e['session'] for e in chosen}), items=ITEMS, chunk=CHUNK,
                   rule='Tier A: train files with an accepted advanced event, round-robin over the 8 advanced milestones (hash order within), '
                        'up to budget-background; Tier B: other train files in hash order up to the budget; calibration/test not read.')
    (args.output / 'selection.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- extraction

SHM = Path('/dev/shm/v7_feature_cache')
CACHE_ROOTS = None
READERS = 2  # system-wide concurrent sequential readers; the HDD array thrashes with more


def staged_copy(entry):
    """Sequentially copy video+actions into RAM under a global reader slot."""
    import fcntl
    import shutil
    SHM.mkdir(exist_ok=True)
    while True:
        for slot in range(READERS):
            lock = open(SHM / f'reader{slot}.lock', 'w')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                video = SHM / f"{entry['stem']}.{os.getpid()}.mp4"
                shutil.copyfile(entry['video'], video)
                raw = Path(entry['actions']).read_bytes()
                return video, raw
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        time.sleep(0.2)


ROOT_WARM, ROOT_WM = 128, 16


def choose_root(entry):
    """One imagination root per fit-session file: a hashed skill-segment start (or any frame if none)."""
    from .v7_data import SKILLS, session_group, skill_labels
    if session_group(entry['session']) != 'fit':
        return None
    labels = skill_labels(entry['frames'], entry['events'])
    starts = [t for t in range(ROOT_WARM, entry['frames'] - 1) if labels[t] != 0 and labels[t - 1] != labels[t]]
    rng = np.random.default_rng(stable_hash('root:' + entry['stem']))
    if starts:
        t0 = int(starts[rng.integers(len(starts))])
    elif entry['frames'] > ROOT_WARM + 1:
        t0 = int(rng.integers(ROOT_WARM, entry['frames'] - 1))
    else:
        return None
    return dict(t0=t0, skill=SKILLS[int(labels[t0])])


def prepare_file(entry):
    import cv2
    video, raw = staged_copy(entry)
    try:
        return decode(entry, video, raw, choose_root(entry))
    finally:
        video.unlink(missing_ok=True)


def decode(entry, video, raw, root=None):
    import cv2
    if hashlib.sha256(raw).hexdigest() != entry['actions_sha256']:
        raise ValueError('Action file changed since audit')
    rows = [json.loads(l) for l in raw.decode('utf-8').splitlines()]
    cursor = cv2.imread(str(VPT_ROOT / 'cursors/mouse_cursor_white_16x16.png'), cv2.IMREAD_UNCHANGED)[:16, :16]
    alpha, cursor = cursor[:, :, 3:] / 255.0, cursor[:, :, :3]
    capture = cv2.VideoCapture(str(video))
    frames, wm_frames = [], []
    for index, row in enumerate(rows):
        ok, frame = capture.read()
        if not ok:
            break
        if root and root['t0'] - ROOT_WM < index <= root['t0']:
            wm_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # WM context: raw frame, no cursor overlay
        if row.get('isGuiOpen'):
            scale = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
            x, y = int(row['mouse']['x'] * scale), int(row['mouse']['y'] * scale)
            h, w = frame.shape[:2]
            ch, cw = max(0, min(16, h - y)), max(0, min(16, w - x))
            if x >= 0 and y >= 0 and ch and cw:
                region = frame[y:y + ch, x:x + cw]
                region[:] = (region * (1 - alpha[:ch, :cw]) + cursor[:ch, :cw] * alpha[:ch, :cw]).astype(np.uint8)
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_LINEAR))
    capture.release()
    count = len(frames)
    if count < len(rows) - 1 or count < 2:
        raise ValueError(f'Video shorter than actions ({count} < {len(rows)})')
    rows = rows[:count]
    if root and len(wm_frames) == ROOT_WM:
        t0 = root['t0']
        target = CACHE_ROOTS / f"{entry['stem']}_{t0}.npz"
        np.savez_compressed(target.with_suffix('.tmp.npz'), vpt_frames=np.stack(frames[t0 - ROOT_WARM + 1:t0 + 1]), wm_frames=np.stack(wm_frames),
                 rows=json.dumps(rows[t0 - ROOT_WM + 1:t0 + 1]), meta=json.dumps(dict(root, stem=entry['stem'], session=entry['session'])))
        target.with_suffix('.tmp.npz').rename(target)
    return rows, np.stack(frames)


def extract(args):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Dataset
    from .vpt_features import load_vpt_policy
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    selection = [json.loads(l) for l in (args.output / 'selection.jsonl').read_text().splitlines()]
    mine = [e for i, e in enumerate(selection) if i % world == rank]
    store = args.output / 'files'
    store.mkdir(exist_ok=True)
    global CACHE_ROOTS
    CACHE_ROOTS = args.output / 'roots'
    CACHE_ROOTS.mkdir(exist_ok=True)
    todo = [e for e in mine if not (store / f"{e['stem']}.npz").exists()]
    policy, mapper, transformer, _ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL),
                                                      weights_sha256=sha256(VPT_WEIGHTS), device=device)

    class Files(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, index):
            entry = todo[index]
            try:
                rows, frames = prepare_file(entry)
            except Exception as exc:
                return entry, None, f'{type(exc).__name__}: {exc}'
            actions = official_actions(rows)
            buttons, camera, null = [], [], []
            for env, is_null in actions:
                factored = transformer.env2policy(env)
                if factored['camera'].ndim == 1:
                    factored = {k: v[None] for k, v in factored.items()}
                agent = mapper.from_factored(factored)
                buttons.append(int(np.asarray(agent['buttons']).reshape(-1)[0]))
                camera.append(int(np.asarray(agent['camera']).reshape(-1)[0]))
                null.append(is_null)
            labels = dict(buttons=np.array(buttons, np.int16), camera=np.array(camera, np.int16), null=np.array(null),
                          gui=np.array([bool(r.get('isGuiOpen')) for r in rows]),
                          inventory=np.array([inventory_row(r) for r in rows], np.int16))
            return entry, frames, labels

    loader = DataLoader(Files(), batch_size=None, collate_fn=lambda item: item, num_workers=args.workers, prefetch_factor=2)
    log = (args.output / f'extract_rank{rank}.jsonl').open('a')
    started, frames_done = time.time(), 0
    with torch.no_grad():
        for n, (entry, frames, labels) in enumerate(loader):
            if frames is None:
                log.write(json.dumps(dict(stem=entry['stem'], status='failed', error=labels)) + '\n')
                log.flush()
                continue
            state = policy.initial_state(1)
            images = torch.from_numpy(frames)
            out = []
            for start in range(0, len(images), CHUNK):
                chunk = images[start:start + CHUNK][None].to(device, non_blocking=True)
                first = torch.zeros(1, chunk.shape[1], dtype=torch.bool, device=device)
                first[0, 0] = start == 0
                (h, _), state = policy.net({'img': chunk}, state, context={'first': first})
                out.append(h[0].half().cpu())
            h = torch.cat(out).numpy()
            if not np.isfinite(h).all():
                log.write(json.dumps(dict(stem=entry['stem'], status='failed', error='nonfinite features')) + '\n')
                continue
            tmp = store / f"{entry['stem']}.h.tmp.npy"
            np.save(tmp, h)
            tmp.rename(store / f"{entry['stem']}.h.npy")
            np.savez(store / f"{entry['stem']}.tmp.npz", events=json.dumps(entry['events']), **labels)
            (store / f"{entry['stem']}.tmp.npz").rename(store / f"{entry['stem']}.npz")
            frames_done += len(h)
            log.write(json.dumps(dict(stem=entry['stem'], status='ok', frames=len(h))) + '\n')
            log.flush()
            if n % 20 == 0:
                rate = frames_done / (time.time() - started)
                print(f'rank {rank} files {n + 1}/{len(todo)} frames/s {rate:.0f}', flush=True)
    print(f'rank {rank} done {frames_done} frames {time.time() - started:.0f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['select', 'extract'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-frames', type=int, default=28_000_000)
    parser.add_argument('--background-frames', type=int, default=6_000_000)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    select(args) if args.mode == 'select' else extract(args)


if __name__ == '__main__':
    main()
