"""V7 Q0 environment probe: observation structure and same-seed reset reproducibility.

Run one process per (seed, repeat) so instance-manager state cannot leak between
resets. The agent takes no-op actions; nothing here evaluates a policy.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JAVA_HOME = os.environ.get('JAVA_HOME', '')
PYTHON = os.environ.get('DREAMER_EVAL_PYTHON', sys.executable)


def structure(value, depth=0):
    if isinstance(value, dict):
        return {k: structure(v, depth + 1) for k, v in sorted(value.items())} if depth < 2 else f'dict[{len(value)}]'
    array = np.asarray(value)
    if array.ndim == 0:
        return f'{array.dtype}:{array.item()!r}'[:60]
    return f'{array.dtype}{list(array.shape)}'


def nonzero_stats(obs):
    out = {}
    for group in ('pickup', 'mine_block', 'craft_item'):
        values = obs.get(group)
        if isinstance(values, dict):
            out[group] = {k: int(np.asarray(v)) for k, v in values.items() if int(np.asarray(v))}
    return out


def noop(env):
    action = env.action_space.no_op()
    action['camera'] = np.zeros(2, dtype=np.float32)
    return action


def probe(seed, steps, out):
    sys.path.insert(0, str(ROOT / 'third_party/Video-Pre-Training'))
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from agent import ENV_KWARGS
    import cv2
    env = HumanSurvival(**ENV_KWARGS).make()
    try:
        env.seed(seed)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        pov = obs['pov']
        cv2.imwrite(str(out / 'reset.png'), cv2.cvtColor(pov, cv2.COLOR_RGB2BGR))
        record = dict(
            seed=seed, keys=sorted(obs), structure={k: structure(v) for k, v in obs.items() if k != 'pov'},
            reset=dict(pov_sha256=hashlib.sha256(pov.tobytes()).hexdigest(), pov_mean=float(pov.mean()),
                       location=structure(obs.get('location_stats', {})), life=structure(obs.get('life_stats', {}))),
        )
        frames = [pov.astype(np.int16)]
        for _ in range(steps):
            result = env.step(noop(env))
            obs = result[0]
            frames.append(obs['pov'].astype(np.int16))
        record['after_noop'] = dict(steps=steps, location=structure(obs.get('location_stats', {})),
                                    life=structure(obs.get('life_stats', {})), stats=nonzero_stats(obs))
        np.save(out / 'frames_small.npy', np.stack([cv2.resize(f.astype(np.uint8), (128, 72)) for f in frames]))
        return record
    finally:
        env.close()


def launch(args):
    base = args.output
    base.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if not JAVA_HOME:
        raise ValueError('Set JAVA_HOME to a Java 8 installation')
    env.update(JAVA_HOME=JAVA_HOME, PATH=f'{JAVA_HOME}/bin:' + env['PATH'], CUDA_VISIBLE_DEVICES='')
    for seed in args.seeds:
        for repeat in range(args.repeats):
            target = base / f'seed{seed}_r{repeat}'
            target.mkdir()
            cmd = ['xvfb-run', '-a', '-s', '-screen 0 1280x720x24', PYTHON, '-m', 'dreamer_agent.q0_env_probe',
                   '--child', '--seed', str(seed), '--steps', str(args.steps), '--output', str(target)]
            with (target / 'log.txt').open('x') as log:
                code = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            print(seed, repeat, 'exit', code, flush=True)
    compare(base)


def compare(base):
    runs = {}
    for path in sorted(base.glob('seed*_r*/record.json')):
        runs[path.parent.name] = json.loads(path.read_text())
    by_seed = {}
    for name, record in runs.items():
        by_seed.setdefault(record['seed'], []).append((name, record))
    report = dict(schema_version=1, scope='v7_q0_env_probe', java_home=JAVA_HOME, runs=len(runs), seeds={})
    for seed, items in by_seed.items():
        frames = [np.load(base / name / 'frames_small.npy') for name, _ in items]
        entry = dict(runs=[name for name, _ in items],
                     reset_pov_sha256=[r['reset']['pov_sha256'] for _, r in items],
                     reset_location=[r['reset']['location'] for _, r in items])
        if len(frames) >= 2:
            n = min(len(f) for f in frames)
            diff = np.abs(frames[0][:n].astype(np.int16) - frames[1][:n].astype(np.int16))
            entry['pixel_mean_abs_diff_by_step'] = [round(float(d.mean()), 3) for d in diff]
        report['seeds'][str(seed)] = entry
    (base / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1)[:4000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[25000, 25001])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if not args.child:
        return launch(args)
    started = time.monotonic()
    record = probe(args.seed, args.steps, args.output)
    record['wall_seconds'] = round(time.monotonic() - started, 1)
    (args.output / 'record.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
