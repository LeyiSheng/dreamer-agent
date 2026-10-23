"""V7 real-environment evaluator (PLAN_V7 §8 Q0, §9).

Pixel-only VPT policies in MineRL HumanSurvival for 5/20/60-minute episodes.
Inventory, stats and life are read only by the isolated scorer below; the policy
input is the resized POV frame. One fresh process per attempt; infrastructure
failures are retried and reported separately, never counted as policy outcomes.

Usage:
  python -m dreamer_agent.eval_long --protocol P.json --output DIR       # run / resume
  python -m dreamer_agent.eval_long --summarize DIR                        # rebuild summary
"""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
FRAME_HZ = 20
SECONDS_PER_6000_STEPS = 517.14  # mean of 40 episodes, outputs/p2_vpt_budget_real_comparison_20260925_retry1
STARTUP_SECONDS = 300
WORLD_SETS = ROOT / 'configs/v7_world_sets.json'
FINAL_LEDGER = ROOT / 'outputs/v7_final_world_usage.jsonl'

WOODS = ('oak', 'birch', 'spruce', 'jungle', 'acacia', 'dark_oak')
LOGS = tuple(f'{w}_log' for w in WOODS) + tuple(f'stripped_{w}_log' for w in WOODS)
PLANKS = tuple(f'{w}_planks' for w in WOODS + ('crimson', 'warped'))
# Milestone -> inventory items whose presence (count >= 1) marks it reached.
MILESTONES = {
    'log': LOGS, 'planks': PLANKS, 'stick': ('stick',), 'crafting_table': ('crafting_table',),
    'wooden_pickaxe': ('wooden_pickaxe',), 'cobblestone': ('cobblestone',), 'stone_pickaxe': ('stone_pickaxe',),
    'furnace': ('furnace',), 'iron_ore': ('iron_ore',), 'iron_ingot': ('iron_ingot',),
    'iron_pickaxe': ('iron_pickaxe',), 'diamond': ('diamond',),
}
TRACKED = sorted({i for items in MILESTONES.values() for i in items})
STAT_KEYS = (('mine_block', 'diamond_ore'), ('pickup', 'diamond'), ('mine_block', 'iron_ore'),
             ('craft_item', 'wooden_pickaxe'), ('craft_item', 'stone_pickaxe'), ('craft_item', 'iron_pickaxe'))


# ---------------------------------------------------------------- scoring

def inventory_counts(obs):
    inventory = obs.get('inventory')
    if not isinstance(inventory, dict):
        raise ValueError('Inventory scorer unavailable')
    return {k.removeprefix('minecraft:'): int(np.asarray(v).sum()) for k, v in inventory.items()}


def stat(obs, group, key):
    values = obs.get(group)
    return int(np.asarray(values.get(key, 0))) if isinstance(values, dict) else 0


class Scorer:
    """Evaluator-only view of inventory/stats. Never passed to the policy."""

    def __init__(self, obs):
        inventory = inventory_counts(obs)
        if any(inventory.get(i, 0) for i in TRACKED):
            raise ValueError('Evaluation requires empty starting inventory')
        self.inventory = {i: 0 for i in TRACKED}
        self.first_hit = {m: None for m in MILESTONES}
        self.diamond_high_water = 0
        self.deaths_start = stat(obs, 'full_stats', 'deaths')
        self.deaths = 0
        self.alive = bool(np.asarray(obs.get('life_stats', {}).get('is_alive', True)))
        self.stats = {f'{g}.{k}': stat(obs, g, k) for g, k in STAT_KEYS}

    def update(self, obs, step):
        """Return a list of change events at this step (for the compact event log)."""
        events = []
        inventory = inventory_counts(obs)
        for item in TRACKED:
            count = inventory.get(item, 0)
            if count != self.inventory[item]:
                events.append(dict(kind='inventory', item=item, before=self.inventory[item], after=count))
                self.inventory[item] = count
        for milestone, items in MILESTONES.items():
            if self.first_hit[milestone] is None and any(self.inventory[i] > 0 for i in items):
                self.first_hit[milestone] = step
                events.append(dict(kind='milestone', milestone=milestone))
        self.diamond_high_water = max(self.diamond_high_water, self.inventory['diamond'])
        deaths = stat(obs, 'full_stats', 'deaths') - self.deaths_start
        alive = bool(np.asarray(obs.get('life_stats', {}).get('is_alive', True)))
        if deaths != self.deaths or alive != self.alive:
            events.append(dict(kind='life', deaths=deaths, alive=alive))
            self.deaths, self.alive = deaths, alive
        for g, k in STAT_KEYS:
            value = stat(obs, g, k)
            if value != self.stats[f'{g}.{k}']:
                events.append(dict(kind='stat', key=f'{g}.{k}', before=self.stats[f'{g}.{k}'], after=value))
                self.stats[f'{g}.{k}'] = value
        return events

    def result(self):
        return dict(first_hit=self.first_hit, diamond_high_water=self.diamond_high_water,
                    diamond_final=self.inventory['diamond'], deaths=self.deaths, stats=self.stats,
                    final_inventory={k: v for k, v in self.inventory.items() if v})


# ---------------------------------------------------------------- one episode (child process)

def location(obs):
    loc = obs.get('location_stats', {})
    return {k: float(np.asarray(loc[k])) for k in ('xpos', 'ypos', 'zpos', 'yaw', 'pitch') if k in loc}


def run_episode(protocol, arm_name, seed, out, port):
    import cv2
    import torch
    from .vpt_features import load_vpt_policy, to_vpt_input
    arm = protocol['arms'][arm_name]
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    # Load installed MineRL before VPT inference-only dependency shims enter sys.path.
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.env.malmo import InstanceManager
    policy, mapper, transformer, dropped = load_vpt_policy(
        ROOT / arm['model']['path'], ROOT / arm['weights']['path'], model_sha256=arm['model']['sha256'],
        weights_sha256=arm['weights']['sha256'], device='cuda:0', allowed_extra_prefixes=tuple(arm.get('extra_prefixes', ())))
    if arm.get('pi_head'):
        head = ROOT / arm['pi_head']['path']
        if sha256(head) != arm['pi_head']['sha256']:
            raise ValueError('pi_head override changed')
        policy.pi_head.load_state_dict(torch.load(head, map_location='cpu', weights_only=True)['pi_head'], strict=True)
    agent = None
    if arm.get('type') == 'v7':
        from .v7_agent import V7Agent
        for key in ('heads', 'skill_policy'):
            if arm.get(key) and sha256(ROOT / arm[key]['path']) != arm[key]['sha256']:
                raise ValueError(f'{key} changed')
        agent = V7Agent(policy, ROOT / arm['heads']['path'],
                        ROOT / arm['skill_policy']['path'] if arm.get('skill_policy') else None, 'cuda:0')
    sys.path.insert(0, str(ROOT / 'third_party/Video-Pre-Training'))
    from agent import ENV_KWARGS
    InstanceManager.configure_malmo_base_port(port)
    env = HumanSurvival(**ENV_KWARGS).make()
    budget = int(protocol['minutes'] * 60 * FRAME_HZ)
    video_cfg = protocol.get('video', dict(fps=2, width=320, height=180))
    try:
        env.seed(seed)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        scorer = Scorer(obs)
        start = dict(pov_sha256=hashlib.sha256(obs['pov'].tobytes()).hexdigest(), location=location(obs))
        # Common random numbers across arms: the sampling stream depends on the world only.
        generator = torch.Generator(device='cuda:0').manual_seed(seed + 901)
        state = policy.initial_state(1)
        video = cv2.VideoWriter(str(out / 'episode.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), float(video_cfg['fps']),
                                (video_cfg['width'], video_cfg['height']))
        video_every = max(1, FRAME_HZ // int(video_cfg['fps']))
        started = time.monotonic()
        done, step, end_reason = False, 0, 'budget'
        skill_steps = {}
        if agent is not None:
            agent.reset()
            last_skill = None
        with torch.no_grad(), (out / 'events.jsonl').open('x') as log:
            for step in range(1, budget + 1):
                if agent is not None:
                    b, c, skill, _ = agent.act(to_vpt_input(obs['pov']), generator)
                    sampled = {'buttons': np.array([[b]]), 'camera': np.array([[c]])}
                    skill_steps[skill] = skill_steps.get(skill, 0) + 1
                    if skill != last_skill:
                        log.write(json.dumps(dict(step=step, kind='skill', skill=skill, ledger=sorted(agent.sched.ledger))) + '\n')
                        last_skill = skill
                else:
                    image = torch.from_numpy(to_vpt_input(obs['pov']).copy())[None, None].cuda()
                    (h, _), state = policy.net({'img': image}, state, context={'first': torch.tensor([[step == 1]], device='cuda')})
                    pd = policy.pi_head(h)
                    sampled = {k: torch.multinomial(pd[k].reshape(-1, pd[k].shape[-1]).float().softmax(-1), 1, generator=generator)
                               .cpu().numpy().reshape(1, 1) for k in ('buttons', 'camera')}
                mapped = transformer.policy2env(mapper.to_factored(sampled))
                action = {k: (np.asarray(v, dtype=np.float32).reshape(2) if k == 'camera' else int(np.asarray(v).reshape(-1)[0]))
                          for k, v in mapped.items()}
                if step % video_every == 1:
                    video.write(cv2.cvtColor(cv2.resize(obs['pov'], (video_cfg['width'], video_cfg['height'])), cv2.COLOR_RGB2BGR))
                result = env.step(action)
                if len(result) == 5:
                    obs, _, terminated, truncated, _ = result
                    done = bool(terminated or truncated)
                else:
                    obs, _, done, _ = result
                for event in scorer.update(obs, step):
                    log.write(json.dumps(dict(step=step, **event)) + '\n')
                if step % 1200 == 0:
                    log.write(json.dumps(dict(step=step, kind='heartbeat', wall=round(time.monotonic() - started, 1),
                                              location=location(obs))) + '\n')
                    log.flush()
                    print('STEP', arm_name, seed, step, scorer.first_hit, flush=True)
                if done:
                    end_reason = 'env_done'
                    break
        video.release()
        return dict(status='ok', arm=arm_name, world_seed=seed, steps=step, budget_steps=budget, end_reason=end_reason,
                    wall_seconds=time.monotonic() - started, start=start, dropped_extra_keys=dropped,
                    skill_steps={str(k): v for k, v in skill_steps.items()}, **scorer.result())
    finally:
        env.close()


def child_main(args):
    protocol = json.loads(args.protocol.read_text())
    out = args.attempt_output
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        result = run_episode(protocol, args.arm, args.seed, out, args.port)
    except Exception as exc:
        result = dict(status='failed', arm=args.arm, world_seed=args.seed, error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
    result['attempt_wall_seconds'] = time.time() - started
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({k: v for k, v in result.items() if k != 'traceback'}), flush=True)


# ---------------------------------------------------------------- protocol and runner

def episode_timeout(minutes, factor):
    return STARTUP_SECONDS + factor * SECONDS_PER_6000_STEPS * (minutes * 60 * FRAME_HZ) / 6000


def validate_protocol(protocol, confirm_final):
    sets = json.loads(WORLD_SETS.read_text())
    world_set = protocol['world_set']
    if world_set not in ('dev', 'final'):
        raise ValueError('world_set must be dev or final')
    if not set(protocol['world_seeds']) <= set(sets[world_set]):
        raise ValueError(f'World seeds not in the registered {world_set} set')
    if world_set == 'final' and not confirm_final:
        raise ValueError('Final (sealed) worlds are only for Q6; pass --confirm-final-q6')
    for arm in protocol['arms'].values():
        for key in ('model', 'weights', 'pi_head', 'heads', 'skill_policy'):
            if key in arm and sha256(ROOT / arm[key]['path']) != arm[key]['sha256']:
                raise ValueError(f'Asset changed: {arm[key]["path"]}')
    java = Path(protocol['runtime']['java_home']) / 'bin/java'
    if not java.exists():
        raise ValueError(f'Java binary missing: {java}')


def jobs(protocol):
    out = []
    for seed in protocol['world_seeds']:
        arms = sorted(protocol['arms'])
        random.Random(seed).shuffle(arms)
        out.extend((arm, seed) for arm in arms)
    return out


def load_attempts(out):
    return [json.loads(p.read_text()) for p in sorted(out.glob('*/attempt_*/result.json'))]


def run(args):
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol, args.confirm_final_q6)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError('Output directory was started with a different protocol')
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if protocol['world_set'] == 'final':
        FINAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with FINAL_LEDGER.open('a') as ledger:
            ledger.write(json.dumps(dict(output=str(out), protocol_sha256=sha256(frozen), started=time.time())) + '\n')
    import fcntl
    lock = (out / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = {(r['arm'], r['world_seed']) for r in load_attempts(out) if r['status'] == 'ok'}
    pending = [j for j in jobs(protocol) if j not in done]
    slots = [(gpu, k) for gpu in protocol['gpus'] for k in range(protocol.get('procs_per_gpu', 1))]
    timeout = episode_timeout(protocol['minutes'], protocol.get('timeout_factor', 1.5))
    runtime = protocol['runtime']
    meta = dict(protocol_sha256=sha256(frozen), runner_sha256=sha256(__file__), pending=len(pending),
                slots=len(slots), timeout_seconds=timeout, started=time.time())
    (out / f'run_{int(time.time())}.json').write_text(json.dumps(meta, indent=2))
    print('RUN', json.dumps(meta), flush=True)
    queue, guard = list(pending), threading.Lock()

    def worker(index, gpu):
        while True:
            with guard:
                if not queue:
                    return
                arm, seed = queue.pop(0)
            job = out / f'{arm}_{seed}'
            job.mkdir(exist_ok=True)
            first = len(list(job.glob('attempt_*/'))) + 1
            for attempt in range(first, protocol.get('max_attempts', 3) + 1):
                target = job / f'attempt_{attempt}'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAVA_HOME=runtime['java_home'],
                           PATH=f"{runtime['java_home']}/bin:" + os.environ['PATH'])
                port = 30000 + index * 1000 + attempt * 100
                cmd = ['xvfb-run', '-a', '-s', '-screen 0 1280x720x24', runtime['python'], '-m', 'dreamer_agent.eval_long',
                       '--child', '--protocol', str(frozen), '--arm', arm, '--seed', str(seed),
                       '--attempt-output', str(target), '--port', str(port)]
                with (job / f'attempt_{attempt}.log').open('x') as log:
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                result_path = target / 'result.json'
                if not result_path.exists():
                    target.mkdir(exist_ok=True)
                    result_path.write_text(json.dumps(dict(status='failed', arm=arm, world_seed=seed,
                                                           error=f'timeout or exit before result (limit {timeout:.0f}s)',
                                                           returncode=process.returncode)))
                status = json.loads(result_path.read_text())['status']
                print('ATTEMPT', arm, seed, attempt, status, flush=True)
                if status == 'ok':
                    break
                time.sleep(10)

    threads = [threading.Thread(target=worker, args=(i, gpu)) for i, (gpu, _) in enumerate(slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summarize(out)


# ---------------------------------------------------------------- summary

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def bootstrap_mean(values, reps=2000, seed=0):
    if not values:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(reps, len(values)), replace=True).mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def summarize(out):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    attempts = load_attempts(out)
    ok = {}
    for r in attempts:
        if r['status'] == 'ok':
            ok.setdefault((r['arm'], r['world_seed']), r)
    expected = jobs(protocol)
    budget = int(protocol['minutes'] * 60 * FRAME_HZ)
    summary = dict(status='complete' if all(j in ok for j in expected) else 'incomplete', world_set=protocol['world_set'],
                   minutes=protocol['minutes'], completed=len(ok), expected=len(expected),
                   missing=[list(j) for j in expected if j not in ok],
                   failed_attempts=sum(r['status'] != 'ok' for r in attempts), arms={})
    for arm in sorted(protocol['arms']):
        rows = [r for (a, _), r in ok.items() if a == arm]
        n = len(rows)
        milestones = {}
        for m in MILESTONES:
            hits = [r['first_hit'][m] for r in rows if r['first_hit'][m] is not None]
            milestones[m] = dict(successes=len(hits), trials=n, wilson95=wilson(len(hits), n),
                                 median_step_when_reached=(float(np.median(hits)) if hits else None),
                                 censored_at_budget=n - len(hits))
        diamonds = [r['diamond_high_water'] for r in rows]
        summary['arms'][arm] = dict(
            episodes=n, milestones=milestones,
            diamond_high_water_mean=(float(np.mean(diamonds)) if diamonds else None),
            diamond_high_water_bootstrap95=bootstrap_mean(diamonds),
            diamond_ore_mined_mean=(float(np.mean([r['stats']['mine_block.diamond_ore'] for r in rows])) if rows else None),
            deaths_mean=(float(np.mean([r['deaths'] for r in rows])) if rows else None),
            ended_before_budget=sum(r['steps'] < budget for r in rows))
    starts = {}
    for (arm, seed), r in ok.items():
        starts.setdefault(seed, {})[arm] = r['start']
    summary['start_state_pairing'] = {
        str(seed): dict(same_pov=len({s['pov_sha256'] for s in v.values()}) == 1,
                        same_location=len({json.dumps(s['location'], sort_keys=True) for s in v.values()}) == 1)
        for seed, v in starts.items()}
    summary['limitations'] = ('Inventory/stats are read only by the scorer. Wilson intervals treat episodes as independent; '
                              'arms share worlds, so compare arms by world-paired analysis where start states match.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1)[:6000])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--confirm-final-q6', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--arm')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--attempt-output', type=Path)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.child:
        child_main(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
