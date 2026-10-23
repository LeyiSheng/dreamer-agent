"""Action-aligned world adapter for V7 (PLAN_V7 §3.1 world-model time index).

Q0 measured (outputs/q0_contract_20260926/action_time) that the action stored in
frame block i changes frame i+1, not frame i: in context, block i holds the
recorded action taken at frame i. FrozenWorldAdapter.step(a) writes `a` into the
block of the frame it generates, so a policy acting on frame t only moves frame
t+2. Here step(a_t) first rewrites block t with a_t using the same context pass
that reset uses for real frames, then generates frame t+1. `a_t` is also stored
provisionally in block t+1 and is overwritten by the next step's action.
"""
import math

import torch

from .world_model import FrozenWorldAdapter


class AlignedWorldAdapter(FrozenWorldAdapter):
    """step(a_t): a_t is the action taken at the latest frame t; returns frame t+1."""

    @torch.inference_mode()
    def _rewrite_last_block(self, action):
        latent = self.latents[-1].to(self.device)
        index = self.frame_index - 1
        maximum = int(self.dynamics.config.K_samples_step)
        noise = torch.randn(latent.shape, device=self.device, dtype=latent.dtype, generator=self.generator)
        with self._autocast():
            self.dynamics(
                img=self.context_signal * latent + (1 - self.context_signal) * noise,
                timestep=torch.tensor([int(maximum * self.context_signal)], device=self.device),
                timestep_stride=torch.tensor([int(math.log2(self.steps)) + 1], device=self.device),
                action_ids=action,
                current_frame_idx=torch.tensor([index], device=self.device),
                read_rope_indices=self.dynamics.get_read_rope_indices(index))
        self.actions[-1] = action.detach().to('cpu', copy=True)

    @torch.inference_mode()
    def step(self, action):
        if not self.ready:
            raise RuntimeError('reset with real context is required')
        self._validate_actions(action, 1)
        try:
            self._rewrite_last_block(action)
        except Exception:
            self.ready = False
            raise
        return super().step(action)
