"""Explicit single-stream orchestration for the verified frozen WM backend.

No old MCWorldModelInfer/prefill wrapper is used. The installed sampler remains
an inference dependency, not an asserted algorithmic ground truth. Caller must
supply already aligned per-observation actions; this module does not guess a
VPT frame offset. Snapshot/restore is serial, not a thread-safe shared backend.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math
import uuid
import weakref
import torch


@dataclass(frozen=True)
class WorldSnapshot:
    owner: str
    frame_index: int
    random_state: torch.Tensor
    caches: dict
    latents: tuple
    actions: tuple


class FrozenWorldAdapter:
    def __init__(self, tokenizer, dynamics, sampler, action_spec, *, seed=0,
                 steps=4, context_signal=.75, image_size=(384, 640)):
        for backend in (tokenizer, dynamics):
            owner = getattr(backend, "_dreamer_adapter_owner", None)
            if owner is not None and owner() is not None:
                raise ValueError("Backend is already owned by a live adapter")
        self.tokenizer = tokenizer.eval().requires_grad_(False)
        self.dynamics = dynamics.eval().requires_grad_(False)
        self.sampler, self.action_spec = sampler, action_spec
        self.device = next(dynamics.parameters()).device
        self.steps, self.context_signal = steps, context_signal
        maximum = int(dynamics.config.K_samples_step)
        if steps < 1 or steps & (steps-1) or steps > maximum//2 or not 0 < context_signal < 1:
            raise ValueError("Unsupported sampling configuration")
        if not float(maximum*context_signal).is_integer():
            raise ValueError("Context signal must lie on the model noise grid")
        self.image_size = tuple(image_size)
        ph, pw = tokenizer.patch_size
        if self.image_size[0] % ph or self.image_size[1] % pw:
            raise ValueError("Image dimensions must match patch geometry")
        self.decode_shape = (1, 1, 3, self.image_size[0]//ph, self.image_size[1]//pw)
        tokenizer.decoder.init_rope_cache(self.decode_shape)
        dynamics.init_rope_cache(frame_seqlen=tokenizer.encoder.num_learnable_tokens//2,
                                 action_slot_len=len(action_spec.slots))
        self.cache_modules = {}
        # Only audited K/V mutable state, never clone model weights or global RNG.
        for prefix, model in (("tokenizer", tokenizer), ("dynamics", dynamics)):
            for name, module in model.named_modules():
                if all(hasattr(module, k) for k in ("k", "v", "clean_cache", "cached_frames")):
                    self.cache_modules[prefix+"."+name] = module
        if not self.cache_modules:
            raise ValueError("Backend has no discoverable audited caches")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.owner = uuid.uuid4().hex
        self.frame_index = 0
        self.latents, self.actions = [], []
        self.ready = False
        self.history_limit = int(dynamics.config.window_size)
        for backend in (tokenizer, dynamics):
            backend._dreamer_adapter_owner = weakref.ref(self)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if self.device.type == "cuda" else nullcontext()

    def _validate_actions(self, actions, length):
        if actions.shape != (length, len(self.action_spec.slots)) or actions.dtype != torch.long:
            raise ValueError("Expected one [slots] action per context frame")
        if actions.device != self.device:
            raise ValueError("Actions must be on the model device")
        for slot, support in enumerate(self.action_spec.slots):
            choices = torch.tensor(support, device=actions.device)
            if not torch.isin(actions[:, slot], choices).all():
                raise ValueError(f"Illegal action token in slot {slot}")

    @torch.inference_mode()
    def clear(self, seed):
        for cache in self.cache_modules.values():
            cache.clean_cache()
            cache.cached_frames = 0
        self.generator.manual_seed(seed)
        self.frame_index = 0
        self.latents, self.actions = [], []
        self.ready = False

    def _remember(self, latent, action):
        self.latents.append(latent.detach().to("cpu", copy=True))
        self.actions.append(action.detach().to("cpu", copy=True))
        self.latents = self.latents[-self.history_limit:]
        self.actions = self.actions[-self.history_limit:]

    @torch.inference_mode()
    def reset(self, pixels, frame_actions, *, seed=0):
        if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] < 1 or tuple(pixels.shape[2:]) != (3, *self.image_size):
            raise ValueError("Expected nonempty [1,time,3,height,width] real context")
        if pixels.device != self.device or not torch.is_floating_point(pixels) or not torch.isfinite(pixels).all():
            raise ValueError("Context must be finite floating pixels on model device")
        if pixels.min() < -1 or pixels.max() > 1:
            raise ValueError("Expected normalized pixels in [-1,1]")
        count = pixels.shape[1]
        if count > min(self.history_limit, int(self.tokenizer.config.window_size)):
            raise ValueError("Context exceeds audited encoder/dynamics window")
        self._validate_actions(frame_actions, count)
        self.clear(seed)
        try:
            with self._autocast():
                # Direct encoder call retains its exception instead of the legacy
                # tokenizer.encode wrapper which swallows RuntimeError.
                encoded = self.tokenizer.encoder(pixels)
                if encoded.ndim != 4 or encoded.shape[:2] != (1, count) or encoded.shape[2] % 2:
                    raise ValueError("Unexpected encoder output geometry")
                tokens = encoded.reshape(1, count, encoded.shape[2]//2, encoded.shape[3]*2)
                if tokens.shape[-1] != self.dynamics.config.in_dim or not torch.isfinite(tokens).all():
                    raise ValueError("Invalid dynamics latent input")
                maximum = int(self.dynamics.config.K_samples_step)
                for index in range(count):
                    # Preserve batch and select exactly one time step in BOTH paths.
                    latent = tokens[:, index:index+1]
                    action = frame_actions[index:index+1]
                    noise = torch.randn(latent.shape, device=self.device, dtype=latent.dtype,
                                        generator=self.generator)
                    self.dynamics(
                        img=self.context_signal*latent+(1-self.context_signal)*noise,
                        timestep=torch.tensor([int(maximum*self.context_signal)], device=self.device),
                        timestep_stride=torch.tensor([int(math.log2(self.steps))+1], device=self.device),
                        action_ids=action,
                        current_frame_idx=torch.tensor([index], device=self.device),
                        read_rope_indices=self.dynamics.get_read_rope_indices(index))
                    reconstructed = self.tokenizer.ar_decode(
                        latent, shape=self.decode_shape,
                        current_frame_idx=torch.tensor([index], device=self.device))
                    if not torch.isfinite(reconstructed).all():
                        raise ValueError("Nonfinite context decoding")
                    self._remember(latent, action)
            self.frame_index = count
            self.ready = True
            return tokens[:, -1:].clone()
        except Exception:
            self.clear(seed)
            raise

    @torch.inference_mode()
    def step(self, action):
        if not self.ready:
            raise RuntimeError("reset with real context is required")
        self._validate_actions(action, 1)
        shape = self.latents[-1].shape
        dtype = next(self.dynamics.parameters()).dtype
        noise = torch.randn(shape, device=self.device, dtype=dtype, generator=self.generator)
        try:
            with self._autocast():
                latent = self.sampler(model=self.dynamics, x=noise, frame_idx=self.frame_index,
                    steps_size=self.steps, action_ids=action, device=self.device,
                    K_samples_max=int(self.dynamics.config.K_samples_step),
                    refresh_kvcache=False, model_runner=None)
                frame = self.tokenizer.ar_decode(latent, shape=self.decode_shape,
                    current_frame_idx=torch.tensor([self.frame_index], device=self.device))
                if not torch.isfinite(latent).all() or not torch.isfinite(frame).all():
                    raise ValueError("Nonfinite generation")
            self._remember(latent, action)
            self.frame_index += 1
            return latent.clone(), frame.clone()
        except Exception:
            self.ready = False  # partially written caches cannot silently be reused
            raise

    @torch.inference_mode()
    def clone_state(self):
        if not self.ready:
            raise RuntimeError("No valid context to snapshot")
        caches = {name: (c.k.to("cpu", copy=True), c.v.to("cpu", copy=True), c.cached_frames)
                  for name, c in self.cache_modules.items()}
        return WorldSnapshot(self.owner, self.frame_index, self.generator.get_state().clone(),
            caches, tuple(t.clone() for t in self.latents), tuple(a.clone() for a in self.actions))

    @torch.inference_mode()
    def restore_state(self, state):
        if state.owner != self.owner or set(state.caches) != set(self.cache_modules):
            raise ValueError("Snapshot belongs to another backend")
        for name, (k, v, _) in state.caches.items():
            c = self.cache_modules[name]
            if k.shape != c.k.shape or v.shape != c.v.shape or k.dtype != c.k.dtype or v.dtype != c.v.dtype:
                raise ValueError("Snapshot cache geometry mismatch")
        self.ready = False
        for name, (k, v, count) in state.caches.items():
            c = self.cache_modules[name]
            c.k.copy_(k)
            c.v.copy_(v)
            c.cached_frames = count
        self.frame_index = state.frame_index
        self.generator.set_state(state.random_state.clone())
        self.latents, self.actions = [t.clone() for t in state.latents], [a.clone() for a in state.actions]
        self.ready = True
