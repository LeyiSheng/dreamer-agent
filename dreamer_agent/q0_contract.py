"""V7 Q0 representation and diamond-metric contract tests (PLAN_V7 §3.1, §4.4).

  python -m dreamer_agent.q0_contract --output outputs/q0_contract_20260926          # CPU + VPT (GPU 0)
  python -m dreamer_agent.q0_contract --output DIR --world                           # + WM action-time probe

Each section returns measurements and a pass flag; the report is hash-bound to the
code and assets it tested.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .diamond_metric import debounced_high_water, high_water
from .eval_long import Scorer
from .provenance import sha256
from .vpt_features import GENERATED_FRAME, REAL_FRAME, generated_to_rgb, load_vpt_policy, to_vpt_input

ROOT = Path(__file__).resolve().parents[1]
VPT_MODEL = ROOT / 'checkpoints/foundation-model-3x.model'
VPT_WEIGHTS = ROOT / 'checkpoints/bc-early-game-3x.weights'
VIDEO = ROOT / 'data/vpt/raw/10xx/shabby-pink-molly-0b427e25d052-20220416-165016.mp4'


# ---------------------------------------------------------------- diamond metric

def fake_obs(diamonds, deaths=0, alive=True):
    return dict(inventory={'diamond': diamonds, 'stick': 0}, full_stats={'deaths': deaths},
                life_stats={'is_alive': alive}, mine_block={}, pickup={}, craft_item={})


def run_scorer(sequence):
    scorer = Scorer(fake_obs(0))
    for step, (count, deaths, alive) in enumerate(sequence, 1):
        scorer.update(fake_obs(count, deaths, alive), step)
    return scorer.result()


def diamond_section():
    cases = {}
    # Evaluation high-water over the true inventory.
    true_cases = dict(
        drop_and_repick=([0, 1, 2, 0, 0, 2, 3], 3),
        consumption_crafting=([0, 1, 2, 3, 0, 0, 1], 3),
        death_drop_respawn=([0, 2, 2, 0, 0, 1], 2),
    )
    for name, (counts, expected) in true_cases.items():
        cases[f'eval_{name}'] = dict(marks=high_water(counts), expected_final=expected, ok=high_water(counts)[-1] == expected)
    seq = [(0, 0, True), (2, 0, True), (0, 1, False), (0, 1, True), (1, 1, True)]
    result = run_scorer(seq)
    cases['eval_scorer_death_respawn'] = dict(result=dict(high_water=result['diamond_high_water'], deaths=result['deaths'],
                                                          first_hit=result['first_hit']['diamond']),
                                              ok=result['diamond_high_water'] == 2 and result['deaths'] == 1 and result['first_hit']['diamond'] == 2)
    # Training high-water over a noisy belief count with debounce m.
    m = 3
    belief_cases = dict(
        single_frame_flicker=([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        two_frame_flicker=([0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        held_increase=([0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]),
        unknown_breaks_hold=([0, 1, None, 1, 1, 1], [0, 0, 0, 0, 0, 1]),
        drop_then_repick_same=([1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0]),
        consumption_then_exceed=([2, 2, 2, 0, 0, 3, 3, 3], [0, 0, 2, 0, 0, 0, 0, 1]),
        death_zero=([1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]),
        jump_by_two=([0, 2, 2, 2], [0, 0, 0, 2]),
    )
    for name, (beliefs, expected) in belief_cases.items():
        _, rewards = debounced_high_water(beliefs, m)
        cases[f'train_{name}'] = dict(rewards=rewards, expected=expected, ok=rewards == expected)
    return dict(definition=dict(eval='high-water of true inventory diamond count', train=f'increment of debounced belief high-water, hold m={m} (m is a Q1 tunable)',
                                note='measures peak simultaneous holding, not cumulative mining'),
                cases=cases, passed=all(c['ok'] for c in cases.values()))


# ---------------------------------------------------------------- preprocessing

def real_frames(count):
    import cv2
    capture = cv2.VideoCapture(str(VIDEO))
    frames = []
    for _ in range(count):
        ok, bgr = capture.read()
        if not ok:
            raise ValueError('Context video too short')
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if frames[0].shape[:2] != REAL_FRAME:
        raise ValueError(f'Unexpected real frame size {frames[0].shape}')
    return frames


def as_generated(frame, dtype):
    """Real uint8 frame -> WM pixel convention ([-1,1], 12+12 rows of 0 = mid-gray) in `dtype`."""
    pixels = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
    pixels = torch.nn.functional.pad(pixels, (0, 0, 12, 12))
    return pixels.to(dtype)[None, None]


def preprocessing_section(frames):
    out = {}
    for label, dtype in (('float32', torch.float32), ('bfloat16', torch.bfloat16)):
        diffs, pixel_off = [], []
        for frame in frames:
            generated = generated_to_rgb(as_generated(frame, dtype))
            crop = generated[12:372]
            pixel_off.append(float((crop != frame).mean()))
            diffs.append(np.abs(to_vpt_input(generated).astype(int) - to_vpt_input(frame).astype(int)))
        stacked = np.stack(diffs)
        out[label] = dict(uint8_roundtrip_fraction_changed=float(np.mean(pixel_off)), vpt_input_max_abs=int(stacked.max()),
                          vpt_input_mean_abs=float(stacked.mean()), vpt_input_mean_signed=float(np.mean([d.mean() for d in diffs])))
    gray = generated_to_rgb(as_generated(frames[0], torch.float32))
    out['pad_rows_value'] = int(np.unique(np.concatenate([gray[:12], gray[372:]]))[0])
    out['contract'] = dict(real='360x640 uint8 RGB -> cv2.resize(128x128, INTER_LINEAR)',
                           to_world='uint8/127.5-1, pad 12+12 rows with 0 in [-1,1] (mid-gray 127), as official Dreamerv4-MC UI',
                           from_world='clamp(-1,1), (x+1)*127.5 rounded to uint8, crop rows 12:372, same resize')
    out['passed'] = out['float32']['vpt_input_max_abs'] <= 1
    return out


# ---------------------------------------------------------------- recurrent state

@torch.no_grad()
def recurrent_section(frames, device):
    params = pickle.load(VPT_MODEL.open('rb'))['model']['args']['net']['args']
    policy, *_ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL), weights_sha256=sha256(VPT_WEIGHTS), device=device)
    images = torch.from_numpy(np.stack([to_vpt_input(f) for f in frames]))[None].to(device)
    count = images.shape[1]

    def stepwise(first_at):
        state, hs = policy.initial_state(1), []
        for t in range(count):
            (h, _), state = policy.net({'img': images[:, t:t + 1]}, state,
                                       context={'first': torch.tensor([[t in first_at]], device=device)})
            hs.append(h[:, 0])
        return torch.stack(hs, 1)

    first = torch.zeros(1, count, dtype=torch.bool, device=device)
    first[0, 0] = True
    (h_seq, _), _ = policy.net({'img': images}, policy.initial_state(1), context={'first': first})
    h_step = stepwise({0})
    half = count // 2
    h_reset = stepwise({0, half})
    state = policy.initial_state(1)
    fresh = []
    for t in range(half, count):
        (h, _), state = policy.net({'img': images[:, t:t + 1]}, state, context={'first': torch.tensor([[t == half]], device=device)})
        fresh.append(h[:, 0])
    fresh = torch.stack(fresh, 1)
    scale = float(h_seq.abs().mean())
    out = dict(
        attention_memory_size=params['attention_memory_size'], timesteps=params['timesteps'],
        attention_mask_style=params['attention_mask_style'], n_recurrence_layers=params['n_recurrence_layers'],
        frames=count, feature_mean_abs=scale,
        stepwise_vs_sequence_max_abs=float((h_step - h_seq).abs().max()),
        first_flag_reset_vs_fresh_max_abs=float((h_reset[:, half:] - fresh).abs().max()),
        history_matters_max_abs=float((h_step[:, half:] - fresh).abs().max()),
        contract=dict(prefill='causal: backbone consumes every real frame from the root file start (first=True only at file start)',
                      memory='clipped causal attention over 256 past frames per layer, 4 layers',
                      file_boundary='first=True at a new file resets memory; verified equal to a fresh start',
                      skill_switch='backbone state is never reset at a skill switch; skill enters only through heads',
                      action_history='VPT backbone takes pixels only; any action input to heads is a separately registered extra'))
    tolerance = 1e-3 * max(scale, 1.0)
    out['passed'] = (out['stepwise_vs_sequence_max_abs'] < tolerance and out['first_flag_reset_vs_fresh_max_abs'] < tolerance
                     and out['history_matters_max_abs'] > 10 * tolerance)
    out['tolerance'] = tolerance
    del policy
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- world-model action time index

def camera_action_index(mapper, transformer, target, buttons):
    for index in range(121):
        env = transformer.policy2env(mapper.to_factored({'buttons': np.array([[buttons]]), 'camera': np.array([[index]])}))
        if np.allclose(np.asarray(env['camera']).reshape(2), target):
            return index
    raise ValueError(f'No camera bin for {target}')


def horizontal_shift(a, b):
    """Horizontal pixel shift of b relative to a (positive: content moved right)."""
    from .q0_camera_probe import shift
    return int(shift(a, b))


@torch.no_grad()
def world_section(device, aligned):
    from .throughput_probe import load_world, real_context
    from .vpt_rl import VPTToWorldAction
    world, spec, codec = load_world(4, aligned=aligned)
    policy, mapper, transformer, _ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL),
                                                     weights_sha256=sha256(VPT_WEIGHTS), device=device)
    del policy
    convert = VPTToWorldAction(mapper, transformer, codec, spec, device)
    # VPT buttons class 0 switches the camera meta-action off; class 1 is camera-only.
    noop = (0, camera_action_index(mapper, transformer, [0, 0], 0))
    left, right = ((1, camera_action_index(mapper, transformer, [0, s * 10], 1)) for s in (-1, 1))

    def tokens(action):
        local = convert(torch.tensor(action, dtype=torch.long))
        return torch.as_tensor(spec.to_tokens(local.cpu().numpy()), device=device, dtype=torch.long)[None]

    frames, pixels, context_tokens = real_context(8, codec)
    world.reset(pixels, context_tokens, seed=0)
    root = world.clone_state()
    last = frames[-1]
    rollouts = {}
    for name, first in (('noop', noop), ('yaw_left', left), ('yaw_right', right)):
        world.restore_state(root)
        seq = []
        for step in range(3):
            _, frame = world.step(tokens(first if step == 0 else noop))
            seq.append(generated_to_rgb(frame)[12:372])
        rollouts[name] = seq
    out = dict(vpt_actions=dict(noop=noop, left=left, right=right), steps={})
    for step in range(3):
        row = {}
        for name in ('yaw_left', 'yaw_right'):
            row[f'{name}_vs_noop_mean_abs'] = float(np.abs(rollouts[name][step].astype(int) - rollouts['noop'][step].astype(int)).mean())
            reference = last if step == 0 else rollouts[name][step - 1]
            row[f'{name}_shift_vs_previous'] = horizontal_shift(reference, rollouts[name][step])
        row['noop_shift_vs_previous'] = horizontal_shift(last if step == 0 else rollouts['noop'][step - 1], rollouts['noop'][step])
        out['steps'][str(step + 1)] = row
    first_row = out['steps']['1']
    out['interpretation'] = ('action passed to step() affects the frame returned by the same call (action enters latent[i])'
                             if min(first_row['yaw_left_vs_noop_mean_abs'], first_row['yaw_right_vs_noop_mean_abs']) > 2.0
                             else 'first returned frame is insensitive to the action; action affects the following frame')
    out['direction_check'] = dict(left_content_moves_right=first_row['yaw_left_shift_vs_previous'] > 8,
                                  right_content_moves_left=first_row['yaw_right_shift_vs_previous'] < -8)
    out['adapter'] = type(world).__name__
    out['passed'] = (out['direction_check']['left_content_moves_right'] and out['direction_check']['right_content_moves_left']
                     and 'same call' in out['interpretation']) if aligned else None
    del world
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frames = real_frames(48)
    report = dict(schema_version=1, scope='v7_q0_contract', code_sha256={p: sha256(ROOT / 'dreamer_agent' / p) for p in
                  ('q0_contract.py', 'vpt_features.py', 'diamond_metric.py', 'eval_long.py')},
                  assets=dict(model=sha256(VPT_MODEL), weights=sha256(VPT_WEIGHTS), video=sha256(VIDEO)))
    report['diamond'] = diamond_section()
    report['preprocessing'] = preprocessing_section(frames)
    report['recurrent'] = recurrent_section(frames, args.device)
    gates = ['diamond', 'preprocessing', 'recurrent']
    if args.world:
        report['world_action_time_legacy'] = world_section(args.device, aligned=False)
        report['world_action_time_aligned'] = world_section(args.device, aligned=True)
        gates.append('world_action_time_aligned')
    report['passed'] = all(report[k]['passed'] for k in gates)
    report['seconds'] = time.time() - started
    name = 'report_world.json' if args.world else 'report.json'
    (args.output / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
