"""V7 VPT backbone loading and the shared pixel preprocessing contract (PLAN_V7 §3.1).

Real MineRL frames (360x640) and Dreamerv4-MC generated frames (384x640) must reach
the VPT backbone through the same function, so train, imagination and deployment
features share one definition.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
VPT_INPUT = (128, 128)
REAL_FRAME = (360, 640)
GENERATED_FRAME = (384, 640)
# Keys present in official RL checkpoints (phasic auxiliary value) that the inference
# policy does not define. Only these may be dropped, and they are always reported.
KNOWN_EXTRA_PREFIXES = ('aux_value_head.',)


def _import_vpt():
    for path in ('third_party/vpt_deps', 'third_party/Video-Pre-Training'):
        entry = str(ROOT / path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def load_vpt_policy(model, weights, *, model_sha256, weights_sha256, device='cpu', allowed_extra_prefixes=()):
    """Load a hash-bound VPT policy; every missing key and every unlisted extra key is an error.

    Disables TF32 process-wide: with it, stepwise and full-sequence backbone features
    differ by up to 0.3 (fp32: 4e-4), breaking the train/imagine/deploy feature contract.
    """
    if sha256(model) != model_sha256 or sha256(weights) != weights_sha256:
        raise ValueError('VPT artifact changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if any(p not in KNOWN_EXTRA_PREFIXES for p in allowed_extra_prefixes):
        raise ValueError('Only reviewed extra-key prefixes may be dropped')
    _import_vpt()
    # .model is pickle: load ONLY the explicitly reviewed hash-bound local asset.
    with Path(model).open('rb') as f:
        params = pickle.load(f)
    from gym3.types import DictType
    from lib.action_mapping import CameraHierarchicalMapping
    from lib.actions import ActionTransformer
    from lib.policy import MinecraftAgentPolicy
    from lib.torch_util import set_default_torch_device
    set_default_torch_device(torch.device(device))
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    options = dict(params['model']['args']['pi_head_opts'])
    options['temperature'] = float(options['temperature'])
    policy = MinecraftAgentPolicy(DictType(**mapper.get_action_space_update()),
                                  params['model']['args']['net']['args'], options).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    expected = set(policy.state_dict())
    dropped = sorted(k for k in state if k not in expected and k.startswith(tuple(allowed_extra_prefixes)))
    policy.load_state_dict({k: v for k, v in state.items() if k not in dropped}, strict=True)
    policy.eval().requires_grad_(False)
    transformer = ActionTransformer(camera_binsize=2, camera_maxval=10, camera_mu=10, camera_quantization_scheme='mu_law')
    return policy, mapper, transformer, dropped


def generated_to_rgb(frame):
    """Decoded WM frame [1,1,3,384,640] in [-1,1] -> 384x640x3 uint8 RGB.

    Rounding quantization: the inverse of real uint8/127.5-1. Truncation (as in the
    legacy rl_pixels.generated_to_p1_pixels) lowers ~80% of pixels by one level.
    """
    if tuple(frame.shape) != (1, 1, 3, *GENERATED_FRAME) or not torch.isfinite(frame).all():
        raise ValueError('Expected one finite decoded frame [1,1,3,384,640]')
    rgb = ((frame[0, 0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return rgb.permute(1, 2, 0).cpu().numpy()


def to_vpt_input(frame):
    """HxWx3 uint8 RGB (real 360x640 or generated 384x640) -> 128x128x3 uint8.

    Direct INTER_LINEAR resize of the full frame, matching official agent.py. Real
    frames enter the WM with 12+12 rows of zero vertical padding, so generated frames
    drop those rows to recover the real 360x640 field of view before resizing.
    """
    import cv2
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected HxWx3 uint8 RGB frame')
    if frame.shape[:2] == GENERATED_FRAME:
        top = (GENERATED_FRAME[0] - REAL_FRAME[0]) // 2
        frame = frame[top:top + REAL_FRAME[0]]
    elif frame.shape[:2] != REAL_FRAME:
        raise ValueError(f'Unsupported frame size {frame.shape[:2]}')
    return cv2.resize(frame, (VPT_INPUT[1], VPT_INPUT[0]), interpolation=cv2.INTER_LINEAR)
