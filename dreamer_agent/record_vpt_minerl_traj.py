"""Record one real MineRL HumanSurvival trajectory with official VPT weights."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VPT_ROOT = ROOT / "third_party" / "Video-Pre-Training"
if str(VPT_ROOT) not in sys.path:
    sys.path.insert(0, str(VPT_ROOT))

MILESTONES = (
    ("log", ("log", "_log")),
    ("planks", ("planks",)),
    ("crafting_table", ("crafting_table",)),
    ("stick", ("stick",)),
    ("wooden_pickaxe", ("wooden_pickaxe",)),
)


def _inventory_names(obs) -> set[str]:
    names = set()
    inv = obs.get("inventory") if isinstance(obs, dict) else None
    if isinstance(inv, dict):
        for k, v in inv.items():
            try:
                count = int(np.asarray(v).sum()) if not np.isscalar(v) else int(v)
            except Exception:
                count = int(bool(v))
            if count > 0:
                names.add(str(k))
    return names


def _hit(names: set[str], patterns) -> bool:
    for name in names:
        low = name.lower()
        for p in patterns:
            if p in low:
                return True
    return False


def _pov(obs) -> np.ndarray:
    pov = obs["pov"] if isinstance(obs, dict) else obs
    arr = np.asarray(pov)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "checkpoints/bc-early-game-3x.weights")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/foundation-model-3x.model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--malmo-base-port", type=int, default=9600)
    a = p.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    max_steps = int(a.seconds * a.fps)

    from agent import ENV_KWARGS, MineRLAgent
    from minerl.env.malmo import InstanceManager
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

    InstanceManager.configure_malmo_base_port(int(a.malmo_base_port))
    print("Making HumanSurvival...", flush=True)
    env = HumanSurvival(**ENV_KWARGS).make()

    params = pickle.load(open(a.model, "rb"))
    net = params["model"]["args"]["net"]["args"]
    pi_head = dict(params["model"]["args"]["pi_head_opts"])
    pi_head["temperature"] = float(pi_head["temperature"])
    agent = MineRLAgent(env, device=a.device, policy_kwargs=net, pi_head_kwargs=pi_head)
    agent.load_weights(str(a.weights))
    print(f"Recording seed={a.seed} steps={max_steps}", flush=True)

    try:
        obs = env.reset(seed=a.seed)
    except TypeError:
        env.seed(a.seed)
        obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    agent.reset()

    frame0 = _pov(obs)
    h, w = frame0.shape[:2]
    raw_mp4 = out / "trajectory.mp4"
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

    first_hit = {name: None for name, _ in MILESTONES}
    actions = []
    t0 = time.time()
    step = 0
    for step in range(max_steps):
        action = agent.get_action(obs)
        # serialize compact action for side-car
        act_rec = {}
        for k, v in action.items():
            if isinstance(v, np.ndarray):
                act_rec[k] = v.tolist()
            else:
                act_rec[k] = v
        actions.append(act_rec)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _r, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _r, done, _info = step_out

        frame = _pov(obs)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        names = _inventory_names(obs)
        wall = time.time() - t0
        for name, patterns in MILESTONES:
            if first_hit[name] is None and _hit(names, patterns):
                first_hit[name] = dict(step=step + 1, seconds=round(wall, 2))
                print(f"HIT {name} @ step={step+1} t={wall:.1f}s", flush=True)
        if (step + 1) % 200 == 0:
            print(f"step {step+1}/{max_steps} wall={wall:.1f}s inv={sorted(names)}", flush=True)
        if done:
            break

    writer.release()
    try:
        env.close()
    except Exception:
        pass

    h264 = out / "trajectory_h264.mp4"
    rc = os.system(f'ffmpeg -y -loglevel error -i "{raw_mp4}" -c:v libx264 -pix_fmt yuv420p "{h264}"')
    video = h264 if rc == 0 and h264.exists() else raw_mp4

    (out / "actions.jsonl").write_text("\n".join(json.dumps(x) for x in actions) + "\n")
    meta = dict(
        schema_version=1,
        kind="vpt_early_game_minerl_trajectory",
        weights=str(a.weights.resolve()),
        model=str(a.model.resolve()),
        seed=a.seed,
        fps=a.fps,
        seconds_requested=a.seconds,
        steps=step + 1,
        wall_seconds=round(time.time() - t0, 2),
        first_hit=first_hit,
        final_inventory=sorted(_inventory_names(obs)),
        video=str(video.resolve()),
        resolution=[int(w), int(h)],
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
