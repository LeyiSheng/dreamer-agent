"""V7 Q0 asset check: hash, strict load and real-frame forward for VPT checkpoints.

Engineering evidence only: a finite forward pass does not show task ability.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256, write_report
from .vpt_features import load_vpt_policy, to_vpt_input

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'bc_early_game_3x': ('checkpoints/foundation-model-3x.model', 'checkpoints/bc-early-game-3x.weights', ()),
    'rl_from_early_game_2x': ('checkpoints/2x.model', 'checkpoints/rl-from-early-game-2x.weights', ('aux_value_head.',)),
}


def read_frames(video, count):
    import cv2
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Video ended after {len(frames)} frames: {video}')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


@torch.no_grad()
def rollout_logits(policy, frames, device):
    state = policy.initial_state(1)
    latents, buttons, camera = [], [], []
    for index, frame in enumerate(frames):
        image = torch.from_numpy(to_vpt_input(frame).copy())[None, None].to(device)
        (pi_h, _), state = policy.net({'img': image}, state, context={'first': torch.tensor([[index == 0]], device=device)})
        pd = policy.pi_head(pi_h)
        latents.append(pi_h.float().cpu())
        buttons.append(pd['buttons'].float().cpu())
        camera.append(pd['camera'].float().cpu())
    return torch.cat(latents, 1), torch.cat(buttons, 1), torch.cat(camera, 1)


def check(name, model, weights, extra, frames, device):
    model_hash, weights_hash = sha256(model), sha256(weights)
    started = time.monotonic()
    policy, _, _, dropped = load_vpt_policy(model, weights, model_sha256=model_hash, weights_sha256=weights_hash,
                                            device=device, allowed_extra_prefixes=extra)
    load_seconds = time.monotonic() - started
    state = torch.load(weights, map_location='cpu', weights_only=True)
    first = rollout_logits(policy, frames, device)
    second = rollout_logits(policy, frames, device)
    latent, buttons, camera = first
    finite = all(bool(torch.isfinite(x).all()) for x in first)
    repeat_max_abs = max(float((a - b).abs().max()) for a, b in zip(first, second))
    report = dict(
        model=dict(path=str(model), sha256=model_hash),
        weights=dict(path=str(weights), sha256=weights_hash, bytes=Path(weights).stat().st_size),
        strict_load='strict after dropping reviewed extra keys' if dropped else True,
        dropped_extra_keys=dropped,
        tensors=len(state),
        parameters=int(sum(v.numel() for v in state.values())),
        heads=sorted({k.split('.')[0] for k in state}),
        latent_dim=int(latent.shape[-1]),
        buttons_classes=int(buttons.shape[-1]),
        camera_classes=int(camera.shape[-1]),
        frames=len(frames),
        finite=finite,
        repeat_max_abs_difference=repeat_max_abs,
        load_seconds=round(load_seconds, 2),
        buttons_top1_first_frames=buttons[0, :8].squeeze(-2).argmax(-1).tolist(),
    )
    del policy
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True, help='Offline train-split video used only as forward input')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    frames = read_frames(args.video, args.frames)
    report = dict(
        schema_version=1, scope='v7_q0_assets', video=dict(path=str(args.video), sha256=sha256(args.video)),
        preprocessing='RGB frame -> cv2.resize(128x128, INTER_LINEAR), same as official agent.py and eval_vpt_rl_paired',
        torch=torch.__version__, assets={},
    )
    for name, (model, weights, extra) in ASSETS.items():
        report['assets'][name] = check(name, ROOT / model, ROOT / weights, extra, frames, args.device)
        print(name, json.dumps(report['assets'][name]), flush=True)
    foundation = ROOT / 'checkpoints/foundation-model-3x.model'
    report['foundation_model_3x_sha256'] = sha256(foundation)
    report['passed'] = all(a['finite'] and a['repeat_max_abs_difference'] == 0 for a in report['assets'].values())
    report['limitations'] = 'Strict load and finite deterministic forward on 16 offline frames; no task-ability claim.'
    write_report(args.output / 'report.json', report)
    print('PASSED' if report['passed'] else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
