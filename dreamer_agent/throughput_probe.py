"""V7 Q0 imagination throughput probe (PLAN_V7 §8 Q0, A1).

Measures one V7 imagination step end to end: VPT policy forward on the current
frame, VPT->WM action conversion, frozen WM sampling + decoding, and conversion of
the generated frame back to the VPT input. Also times clone/restore used by risk
branches, and aggregate throughput with several processes sharing one GPU.
Throughput evidence only: no reward, no learning, no semantic claim.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / 'data/vpt/raw/10xx/shabby-pink-molly-0b427e25d052-20220416-165016.mp4'
VPT_MODEL = ROOT / 'checkpoints/foundation-model-3x.model'
VPT_WEIGHTS = ROOT / 'checkpoints/bc-early-game-3x.weights'


def load_world(steps, aligned=False):
    from .actions import ActionSpec
    from .world_model import FrozenWorldAdapter
    from .world_v7 import AlignedWorldAdapter
    assets = json.loads((ROOT / 'outputs/acceptance_20260913/assets.json').read_text())
    for entry in assets['files']:
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('WM assets changed')
    snapshot = Path(assets['files'][0]['path']).parents[1]
    sys.path.insert(0, str(ROOT / 'Dreamerv4-MC'))
    from src.modules.tokenizer import CausalTokenizer
    from src.modules.dynamic_model import DynamicModel
    from src.modules.actokenizer import MineCraftActionTokenizer
    from src.inference.infer import generate_one_frame
    models = []
    for cls, sub in ((CausalTokenizer, 'tokenizer'), (DynamicModel, 'dynamic')):
        net, info = cls.from_pretrained(str(snapshot), subfolder=sub, local_files_only=True,
                                        torch_dtype=torch.bfloat16, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Strict WM loading failed')
        models.append(net.cuda().eval().requires_grad_(False))
    spec = ActionSpec.from_dict(json.loads((ROOT / 'outputs/acceptance_20260913/action_probe.json').read_text())['action_spec'])
    world = (AlignedWorldAdapter if aligned else FrozenWorldAdapter)(*models, generate_one_frame, spec, seed=0, steps=steps)
    return world, spec, MineCraftActionTokenizer()


def real_context(count, codec):
    import cv2
    capture = cv2.VideoCapture(str(VIDEO))
    frames = []
    for _ in range(count):
        ok, bgr = capture.read()
        if not ok:
            raise ValueError('Context video too short')
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    rows = [json.loads(line) for line in VIDEO.with_suffix('.jsonl').read_text().splitlines()[:count]]
    tokens = [codec.get_action_index_from_actiondict(codec.json_action_to_env_action(r, hotbar=True)[0], include_gui=True) for r in rows]
    pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
    pixels = torch.nn.functional.pad(pixels, (0, 0, 12, 12))[None].cuda()
    return frames, pixels, torch.tensor(tokens, dtype=torch.long, device='cuda')


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def summarize(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values), p90=sorted(values)[int(0.9 * (len(values) - 1))])


@torch.no_grad()
def measure(args):
    from .vpt_features import load_vpt_policy, generated_to_rgb, to_vpt_input
    from .vpt_rl import VPTToWorldAction
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    world, spec, codec = load_world(args.sampling_steps, aligned=args.aligned)
    vpt, mapper, transformer, _ = load_vpt_policy(VPT_MODEL, VPT_WEIGHTS, model_sha256=sha256(VPT_MODEL),
                                                  weights_sha256=sha256(VPT_WEIGHTS), device='cuda:0')
    convert = VPTToWorldAction(mapper, transformer, codec, spec, 'cuda:0')
    generator = torch.Generator(device='cuda:0').manual_seed(args.seed)
    report = dict(sampling_steps=args.sampling_steps, aligned=args.aligned, context=args.context, rollout=args.rollout,
                  weights_after_load_bytes=torch.cuda.memory_allocated())

    frames, pixels, tokens = real_context(args.context, codec)
    _, report['prefill_seconds'] = timed(lambda: world.reset(pixels, tokens, seed=args.seed))
    state = vpt.initial_state(1)
    for index, frame in enumerate(frames):
        image = torch.from_numpy(to_vpt_input(frame).copy())[None, None].cuda()
        (h, _), state = vpt.net({'img': image}, state, context={'first': torch.tensor([[index == 0]], device='cuda')})

    parts = {k: [] for k in ('policy', 'convert', 'world', 'to_vpt', 'backbone', 'total')}
    for _ in range(args.rollout):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        pd, t_policy = timed(lambda: vpt.pi_head(h))
        action = torch.stack([torch.multinomial(pd[k].reshape(-1, pd[k].shape[-1]).float().softmax(-1), 1, generator=generator).reshape(())
                              for k in ('buttons', 'camera')])
        local, t_convert = timed(lambda: convert(action))
        world_tokens = torch.as_tensor(spec.to_tokens(local.cpu().numpy()), device='cuda', dtype=torch.long)[None]
        (_, frame), t_world = timed(lambda: world.step(world_tokens))
        image, t_to_vpt = timed(lambda: torch.from_numpy(to_vpt_input(generated_to_rgb(frame)).copy())[None, None].cuda())
        ((h, _), state), t_backbone = timed(lambda: vpt.net({'img': image}, state, context={'first': torch.tensor([[False]], device='cuda')}))
        torch.cuda.synchronize()
        for key, value in zip(parts, (t_policy, t_convert, t_world, t_to_vpt, t_backbone, time.perf_counter() - begin)):
            parts[key].append(value)
    warm = args.warmup
    report['step_seconds'] = {k: summarize(v[warm:]) for k, v in parts.items()}
    report['transitions_per_second'] = 1 / report['step_seconds']['total']['mean']

    snapshot, t_clone = timed(world.clone_state)
    _, t_restore = timed(lambda: world.restore_state(snapshot))
    report['clone_seconds'], report['restore_seconds'] = t_clone, t_restore
    report['snapshot_bytes'] = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v, _ in snapshot.caches.values())
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['frame_index_end'] = world.frame_index
    return report


def launch(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    configs = [dict(steps=s, procs=1) for s in args.sampling_list] + [dict(steps=args.sampling_list[0], procs=p) for p in args.procs_list if p > 1]
    results = []
    for config in configs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1')
        children = []
        for proc in range(config['procs']):
            target = out / f"steps{config['steps']}_procs{config['procs']}_{proc}.json"
            cmd = [sys.executable, '-m', 'dreamer_agent.throughput_probe', '--child', '--result', str(target),
                   '--sampling-steps', str(config['steps']), '--context', str(args.context), '--rollout', str(args.rollout),
                   '--warmup', str(args.warmup), '--seed', str(proc)] + (['--aligned'] if args.aligned else [])
            log = (out / (target.stem + '.log')).open('x')
            children.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), target))
        for process, _ in children:
            process.wait()
        rows = [json.loads(t.read_text()) for _, t in children if t.exists()]
        entry = dict(config, completed=len(rows), per_process_tps=[r['transitions_per_second'] for r in rows])
        entry['aggregate_tps'] = sum(entry['per_process_tps'])
        results.append(entry)
        print(json.dumps(entry), flush=True)
    base = next((r for r in results if r['procs'] == 1 and r['steps'] == args.sampling_list[0]), None)
    report = dict(schema_version=1, scope='v7_q0_throughput', gpu=args.gpu, aligned=args.aligned, results=results,
                  sample_budget_threshold_24h=1_000_000,
                  note='transitions_per_second excludes reward heads (V7 heads are small MLPs on h) and K risk branches; '
                       'K branches multiply WM cost for the steps where risk is evaluated.')
    best = max((r['aggregate_tps'] for r in results if r['completed'] == r['procs']), default=0)
    report['best_aggregate_tps_one_gpu'] = best
    report['transitions_24h_one_gpu'] = best * 86400
    report['legacy_8h_run_tps'] = 118848 / 28800
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sampling-list', type=int, nargs='+', default=[4, 2])
    parser.add_argument('--procs-list', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--sampling-steps', type=int, default=4)
    parser.add_argument('--context', type=int, default=16)
    parser.add_argument('--rollout', type=int, default=96)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--aligned', action='store_true')
    args = parser.parse_args()
    if not args.child:
        return launch(args)
    try:
        report = measure(args)
    except Exception as exc:
        import traceback
        report = dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        args.result.with_suffix('.error.json').write_text(json.dumps(report, indent=2))
        raise
    args.result.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
