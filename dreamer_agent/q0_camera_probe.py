"""V7 Q0 real-environment camera gain probe (compare with WM action-time probe).

Applies yaw +/-10 degree camera actions in MineRL HumanSurvival and measures the
horizontal pixel shift of the next observations, with the same estimator used for
generated frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def shift(a, b):
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[60:300].astype(float)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)[60:300].astype(float)
    return min(range(-160, 161), key=lambda d: np.abs(ga[:, max(0, -d):640 - max(0, d)] - gb[:, max(0, d):640 - max(0, -d)]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=30001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    sys.path.insert(0, str(ROOT / 'third_party/Video-Pre-Training'))
    from agent import ENV_KWARGS
    env = HumanSurvival(**ENV_KWARGS).make()
    try:
        env.seed(args.seed)
        obs = env.reset()
        noop = env.action_space.noop()
        for _ in range(40):
            obs, *_ = env.step(noop)
        plans = {'right_once': [10, 0, 0, 0], 'left_once': [-10, 0, 0, 0], 'right_hold': [10, 10, 10, 10], 'left_hold': [-10, -10, -10, -10]}
        results = {}
        for name, yaws in plans.items():
            shifts, previous = [], obs['pov']
            for yaw in yaws:
                action = dict(noop, camera=np.array([0.0, float(yaw)], dtype=np.float32))
                obs, *_ = env.step(action)
                shifts.append(shift(previous, obs['pov']))
                previous = obs['pov']
            for _ in range(10):
                obs, *_ = env.step(noop)
            results[name] = shifts
    finally:
        env.close()
    report = dict(schema_version=1, scope='v7_q0_real_camera_gain', seed=args.seed, shift_px_vs_previous=results,
                  note='positive yaw turns right; content then shifts left (negative px)')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
