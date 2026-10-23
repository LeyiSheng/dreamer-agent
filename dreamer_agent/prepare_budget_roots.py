"""Freeze diverse offline train roots for the authorized VPT proxy experiment."""
import argparse
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

from .provenance import sha256


def candidates(audit, seed):
    by_session = {}
    splits = defaultdict(set)
    with Path(audit).open() as src:
        for line in src:
            r = json.loads(line)
            group = r['source_session'].rsplit('-', 1)[-1]
            splits[group].add(r['split'])
            if r['split'] != 'train' or r['status'] != 'eligible_pending_full_decode':
                continue
            events = [e['frame'] for e in r['events'] if e['task'] == 'mine_log' and e['accepted']]
            if not events or r['frames'] < 32:
                continue
            rank = hashlib.sha256(f"{seed}:{r['stem']}".encode()).hexdigest()
            item = {k: r[k] for k in ('version', 'stem', 'identity', 'source_session', 'video_path', 'actions_path', 'actions_sha256', 'frames')}
            item.update(group=group, rank=rank, event_frames=events)
            if group not in by_session or rank < by_session[group]['rank']:
                by_session[group] = item
    buckets = defaultdict(lambda: defaultdict(list))
    for group, r in by_session.items():
        if splits[group] == {'train'}:
            buckets[r['version']][r['identity']].append(r)
    rng = random.Random(seed)
    per_version = {}
    for version, identities in sorted(buckets.items()):
        order = sorted(identities); rng.shuffle(order)
        for name in order:
            identities[name].sort(key=lambda x: x['rank'])
        queue = deque(order); rows = []
        while queue:
            name = queue.popleft()
            rows.append(identities[name].pop())
            if identities[name]: queue.append(name)
        per_version[version] = deque(rows)
    versions = deque(sorted(per_version)); result = []
    while versions:
        version = versions.popleft()
        result.append(per_version[version].popleft())
        if per_version[version]: versions.append(version)
    return result


def extract(r, output, seed):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    try:
        if sha256(r['actions_path']) != r['actions_sha256']:
            raise ValueError('Action source changed after audit')
        video_hash = sha256(r['video_path'])
        rows = [json.loads(l) for l in Path(r['actions_path']).read_text().splitlines()]
        rng = random.Random(f"{seed}:{r['stem']}")
        event = rng.choice(r['event_frames'])
        relevant = max(0, event - 8)
        uniform = rng.randrange(len(rows) - 16)
        if uniform == relevant: uniform = (uniform + 1) % (len(rows) - 16)
        wanted = {relevant, uniform}; frames = {}
        cap = cv2.VideoCapture(r['video_path'])
        try:
            for index in range(max(wanted) + 1):
                ok, frame = cap.read()
                if not ok: raise ValueError(f'Prefix decode failed at frame {index}')
                if index in wanted:
                    if frame.shape != (360, 640, 3): raise ValueError('Unexpected root image shape')
                    frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        if sha256(r['video_path']) != video_hash or sha256(r['actions_path']) != r['actions_sha256']:
            raise ValueError('Source changed during extraction')
        entries = []
        for role, index in [('event_related', relevant), ('uniform', uniform)]:
            path = Path(output) / f"{r['stem']}_{role}.npz"
            np.savez_compressed(path, rgb=frames[index])
            entries.append(dict(stem=r['stem'], source_session=r['source_session'], session_group=r['group'],
                version=r['version'], identity_alias=r['identity'], frame=index, role=role,
                selection_event_frame=event if role == 'event_related' else None,
                video=dict(path=r['video_path'], sha256=video_hash),
                actions=dict(path=r['actions_path'], sha256=r['actions_sha256']),
                image=dict(path=str(path.resolve()), sha256=sha256(path)), action_row=rows[index],
                source_split='train', source_origin='offline', decoded_prefix_frames=max(wanted)+1))
        return dict(status='ok', entries=entries)
    except Exception as exc:
        return dict(status='rejected', stem=r['stem'], error=f'{type(exc).__name__}: {exc}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sessions', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260922)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    audit_hash = sha256(a.audit)
    pool = candidates(a.audit, a.seed)
    roots, rejected, seen_actions, seen_videos = [], [], set(), set()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(pool), 4):
            batch = pool[start:start+min(4, a.sessions-len(roots)//2)]
            if not batch: break
            results = list(workers.map(lambda r: extract(r, a.output, a.seed), batch))
            for result in results:
                if result['status'] != 'ok': rejected.append(result); continue
                entry = result['entries'][0]
                ah, vh = entry['actions']['sha256'], entry['video']['sha256']
                if ah in seen_actions or vh in seen_videos:
                    rejected.append(dict(status='duplicate', stem=entry['stem'])); continue
                seen_actions.add(ah); seen_videos.add(vh); roots.extend(result['entries'])
            print(json.dumps(dict(sessions=len(roots)//2, roots=len(roots), rejected=len(rejected))), flush=True)
            if len(roots)//2 == a.sessions: break
    if len(roots)//2 != a.sessions: raise RuntimeError('Insufficient independently grouped train sources')
    if sha256(a.audit) != audit_hash: raise RuntimeError('Audit changed')
    manifest = dict(schema_version=1, scope='offline_train_proxy_roots', selection_seed=a.seed,
        audit=dict(path=str(a.audit.resolve()), sha256=audit_hash), roots=roots, rejected=rejected,
        sessions=a.sessions, source_versions=dict(Counter(r['version'] for r in roots[::2])),
        identity_aliases=len({r['identity_alias'] for r in roots}),
        limitations=['Prefix through both roots decoded; unconsumed video tail is not certified.',
                    'Uniform roots are not labelled negative or successful.',
                    'Same-index action conditioning is diagnostic, not physically calibrated.',
                    'No evaluation trajectories used; near-duplicate imagery not fully certified.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', json.dumps({k:v for k,v in manifest.items() if k not in ('roots','rejected')}), flush=True)


if __name__ == '__main__': main()
