"""VPT pretrained readout RL, preserving the official frozen recurrent backbone.

This is a separately declared P2 variant: VPT buttons/camera, one-step BC, no
invented MTP heads. Reward remains on its own frozen P1 feature/history branch.
Only inference architecture is imported; no upstream training code is reused.
"""
from copy import deepcopy
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .provenance import sha256
from .imagination import ImaginationTrainer


class VPTReadout(nn.Module):
    targets = 1
    keys = ('buttons', 'camera')

    def __init__(self, policy):
        super().__init__()
        self.head = deepcopy(policy.pi_head)
        self.action_sizes = tuple(self.head[k].num_actions for k in self.keys)
        if any(tuple(self.head[k].output_shape[:-1]) != (1,) for k in self.keys):
            raise ValueError('Expected one categorical entry per VPT factor')

    def forward(self, history, tasks, *, all_targets=False):
        if (tasks != 0).any():
            raise ValueError('Initial VPT experiment supports one fixed task only')
        pd = self.head(history.detach())
        scores = [pd[k].squeeze(-2) for k in self.keys]
        return [x.unsqueeze(-2) for x in scores] if all_targets else scores


def load_vpt(model, weights, *, model_sha256, weights_sha256, device='cpu'):
    if sha256(model) != model_sha256 or sha256(weights) != weights_sha256:
        raise ValueError('VPT artifact changed')
    root = Path(__file__).resolve().parents[1]
    for path in ('third_party/vpt_deps', 'third_party/Video-Pre-Training'):
        sys.path.insert(0, str(root / path))
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
    policy.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    policy.eval().requires_grad_(False)
    transformer = ActionTransformer(camera_binsize=2, camera_maxval=10, camera_mu=10, camera_quantization_scheme='mu_law')
    return policy, mapper, transformer


class VPTFeatureBackend:
    """Independent VPT pixel history plus existing frozen reward/WM bridge.

    reward_backend must implement the reviewed single-task-origin P1 contract.
    vpt_to_local must reject unsupported codec conversions (never clip/fill).
    """
    def __init__(self, reward_backend, vpt, vpt_to_local):
        self.reward_backend, self.vpt, self.vpt_to_local = reward_backend, vpt, vpt_to_local
        self.frozen_modules = (*reward_backend.frozen_modules, vpt)

    @torch.no_grad()
    def _observe(self, pixels, first):
        import cv2
        # Preserve official uint8 RGB resize protocol, cropping WM padding.
        rgb = ((pixels[0,0,:,12:-12].float().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
        image = cv2.resize(rgb, (128,128), interpolation=cv2.INTER_LINEAR)
        device = next(self.vpt.parameters()).device
        obs = {'img': torch.from_numpy(image.copy())[None,None].to(device)}
        (h, _), self.state = self.vpt.net(obs, self.state,
            context={'first': torch.tensor([[first]], device=device)})
        return h[0,0].detach().clone()

    def reset(self, root, task, seed):
        if task != 0:
            raise ValueError('Single fixed VPT task required')
        self.reward_backend.reset(root, task, seed)
        self.state = self.vpt.initial_state(1)
        # The final context observation is consumed exactly once, not twice.
        return self._observe(root.pixels, True)

    def step(self, action):
        local = self.vpt_to_local(action)
        result = self.reward_backend.step(local)
        if not result.uncertainty_cut:
            result.history = self._observe(self.reward_backend.last_pixels, False)
        return result


def make_trainer(vpt, backend, config, threshold, *, seed, data_binding):
    return ImaginationTrainer(vpt, backend, config, {0: threshold}, seed=seed,
        data_binding=data_binding, policy_readout=VPTReadout(vpt),
        history_dim=vpt.net.output_latent_size(), tasks=1)


class VPTToWorldAction:
    """Explicit conversion; no camera clipping or missing-button substitution."""
    def __init__(self, mapper, transformer, codec, spec, device):
        self.mapper, self.transformer, self.codec, self.spec, self.device = mapper, transformer, codec, spec, device

    def __call__(self, action):
        if action.shape != (2,) or action.dtype != torch.long:
            raise ValueError('Expected VPT buttons/camera categorical action')
        pd = {k: np.array([[int(action[i])]],dtype=np.int64) for i,k in enumerate(VPTReadout.keys)}
        if not (0 <= int(action[0]) < 8641 and 0 <= int(action[1]) < 121):
            raise ValueError('Action outside verified 3x VPT support')
        env = self.transformer.policy2env(self.mapper.to_factored(pd))
        flat = {k: (np.asarray(v,dtype=np.float32).reshape(2) if k == 'camera' else int(np.asarray(v).reshape(-1)[0]))
                for k,v in env.items()}
        if 'camera' not in flat or not np.isfinite(flat['camera']).all():
            raise ValueError('Invalid VPT camera action')
        tokens = self.codec.get_action_index_from_actiondict(flat, include_gui=True)
        local = self.spec.to_local(np.asarray(tokens,dtype=np.int64))
        return torch.tensor(local,device=self.device,dtype=torch.long)
