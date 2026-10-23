"""Minimal real-env eval for official VPT early-game BC (log -> wooden pickaxe).

Requires MineRL + Java. Runs N episodes under Xvfb, reports first-obtain
success rates and times for early-game milestones. Does not tune checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VPT_ROOT = ROOT / "third_party" / "Video-Pre-Training"
for path in (str(VPT_ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)

# Milestone item name substrings matched against MineRL inventory / equipped items.
MILESTONES = (
    ("log", ("log", "_log")),
    ("planks", ("planks",)),
    ("crafting_table", ("crafting_table",)),
    ("stick", ("stick",)),
    ("wooden_pickaxe", ("wooden_pickaxe",)),
    ("cobblestone", ("cobblestone",)),
    ("stone_pickaxe", ("stone_pickaxe",)),
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
    # Some MineRL builds expose equipped item separately.
    for key in ("equipped_items", "equipped_item"):
        eq = obs.get(key) if isinstance(obs, dict) else None
        if isinstance(eq, dict):
            main = eq.get("mainhand") or eq.get("main_hand")
            if isinstance(main, dict) and main.get("type"):
                names.add(str(main["type"]))
            elif isinstance(main, str):
                names.add(main)
    return names


def _hit(names: set[str], patterns) -> bool:
    for name in names:
        low = name.lower()
        for p in patterns:
            if p in low:
                return True
    return False


def _load_agent(model, weights, env, device):
    from agent import MineRLAgent

    params = pickle.load(open(model, "rb"))
    net = params["model"]["args"]["net"]["args"]
    pi_head = dict(params["model"]["args"]["pi_head_opts"])
    pi_head["temperature"] = float(pi_head["temperature"])
    agent = MineRLAgent(env, device=device, policy_kwargs=net, pi_head_kwargs=pi_head)
    agent.load_weights(str(weights))
    return agent


def run_episode(env, agent, *, max_steps: int, seed: int | None):
    if seed is not None:
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            # Older gym API
            env.seed(seed)
            obs = env.reset()
    else:
        obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    agent.reset()
    first_hit = {name: None for name, _ in MILESTONES}
    t0 = time.time()
    for step in range(max_steps):
        action = agent.get_action(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out
        names = _inventory_names(obs)
        wall = time.time() - t0
        for name, patterns in MILESTONES:
            if first_hit[name] is None and _hit(names, patterns):
                first_hit[name] = dict(step=step + 1, seconds=round(wall, 2))
        if done:
            break
    return dict(
        steps=step + 1,
        wall_seconds=round(time.time() - t0, 2),
        first_hit=first_hit,
        final_inventory=sorted(_inventory_names(obs)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "checkpoints/bc-early-game-3x.weights")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/foundation-model-3x.model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-minutes", type=float, default=10.0, help="Per-episode wall-clock/step budget at 20 FPS")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument(
        "--malmo-base-port",
        type=int,
        default=None,
        help="MineRL/Malmo base TCP port (set disjoint values when multi-GPU parallel)",
    )
    p.add_argument("--smoke", action="store_true", help="1 episode, 30s budget")
    a = p.parse_args()

    if a.smoke:
        a.episodes = 1
        a.max_minutes = 0.5

    max_steps = int(a.max_minutes * 60 * a.fps)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)

    # Official VPT env kwargs from agent.py
    from agent import ENV_KWARGS
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.env.malmo import InstanceManager

    if a.malmo_base_port is not None:
        InstanceManager.configure_malmo_base_port(int(a.malmo_base_port))
        print(f"Malmo base port={a.malmo_base_port}", flush=True)

    print("Making HumanSurvival env...", flush=True)
    env = HumanSurvival(**ENV_KWARGS).make()
    print("Loading VPT agent...", flush=True)
    agent = _load_agent(a.model, a.weights, env, a.device)

    records = []
    for i in range(a.episodes):
        seed = a.base_seed + i
        print(f"EPISODE {i+1}/{a.episodes} seed={seed}", flush=True)
        try:
            rec = run_episode(env, agent, max_steps=max_steps, seed=seed)
            rec.update(episode=i, seed=seed, status="ok")
        except Exception as exc:
            rec = dict(episode=i, seed=seed, status="failed", error=f"{type(exc).__name__}: {exc}")
            print("FAILED", rec["error"][:200], flush=True)
        records.append(rec)
        with (out / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # Aggregate success rates
    ok = [r for r in records if r.get("status") == "ok"]
    summary = {
        "schema_version": 1,
        "kind": "vpt_early_game_minerl_eval",
        "weights": str(a.weights.resolve()),
        "model": str(a.model.resolve()),
        "episodes_requested": a.episodes,
        "episodes_ok": len(ok),
        "max_minutes": a.max_minutes,
        "max_steps": max_steps,
        "fps": a.fps,
        "base_seed": a.base_seed,
        "device": a.device,
        "milestones": {},
        "note": "Real MineRL HumanSurvival; success = item appears in inventory at least once within budget.",
    }
    for name, _ in MILESTONES:
        hits = [r["first_hit"][name] for r in ok if r.get("first_hit", {}).get(name)]
        times = [h["seconds"] for h in hits]
        summary["milestones"][name] = {
            "successes": len(hits),
            "trials": len(ok),
            "success_rate": (len(hits) / len(ok)) if ok else None,
            "mean_seconds_when_success": (float(np.mean(times)) if times else None),
            "median_seconds_when_success": (float(np.median(times)) if times else None),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
