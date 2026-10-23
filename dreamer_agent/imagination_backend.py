"""Bridge frozen world adapter to the P1 single-frame feature/history interface.

encode_pixels takes one padded normalized RGB [1,1,3,384,640] frame and returns
[feature_dim], using the SAME independent-frame extractor as P1. It must not
flatten a generated dynamics latent as a substitute for re-encoding pixels.
Frame/action alignment and pixel conversion require real integration acceptance.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .contracts import Episode
from .imagination import ModelStep


@dataclass
class OfflineRoot:
    episode: Episode
    pixels: torch.Tensor
    frame_actions: torch.Tensor  # Explicit audited WM conditioning convention.
    features: torch.Tensor  # Verified matching independent-frame P1 features.
    provenance: dict
    past_actions: torch.Tensor  # Local one-hot a_<t; known previous action retained.


class FrozenFeatureBackend:
    def __init__(self, world, bc, reward, encode_pixels, frame_to_pixels, *, consistency,
                 context=None, risk=None, contract=None, policy_metadata=None, reward_metadata=None, outcome=None):
        self.world, self.bc, self.reward = world, bc, reward
        self.encode_pixels, self.frame_to_pixels = encode_pixels, frame_to_pixels
        self.consistency, self.risk, self.outcome = consistency, risk, outcome
        self.contract, self.readout = contract, None
        self.context = (32 if contract is None else contract.context) if context is None else context
        self.frozen_modules = (world.tokenizer, world.dynamics, bc, reward)
        if contract is None:
            if self.context != 32:
                raise ValueError('Nonlegacy context requires a verified history/reward contract')
        else:
            from .rl_contract import StreamingReadout
            if self.context != contract.context or world.action_spec.fingerprint != contract.action_spec_sha256:
                raise ValueError('Runtime history/action contract mismatch')
            contract.verify_training_metadata(policy_metadata or {}, reward_metadata or {})
            self.readout = StreamingReadout(bc, reward, contract)
        self.ready = False

    @torch.no_grad()
    def reset(self, root, task, seed):
        self.ready = False
        if root.episode.origin != 'offline' or root.episode.split != 'train':
            raise ValueError('Only offline train roots allowed')
        from .rl_roots import verify_root
        manifest = verify_root(root)
        if self.contract is not None:
            self.contract.verify_root(manifest)
            self.contract.reward_task(task)
        n = len(root.features)
        if n != 1 or root.pixels.shape[1] != n or len(root.past_actions) != n:
            raise ValueError('Root must start at a known BC block boundary, with matching context')
        if torch.count_nonzero(root.past_actions):
            raise ValueError('Task-start BC past action must be zero as in P1')
        actual = self.encode_pixels(root.pixels).detach().to(root.features)
        if not torch.equal(actual.reshape(-1), root.features[0]):
            raise ValueError('Root feature does not match P1 pixel encoder')
        self.world.reset(root.pixels, root.frame_actions, seed=seed)
        self.last_pixels = root.pixels.detach().clone()
        self.last_feature = root.features[-1].detach().clone()
        self.features = [x.detach().clone() for x in root.features]
        self.past = [x.detach().clone() for x in root.past_actions]
        self.task = task
        # Reward completion memory begins at the root's final real observation.
        # No pre-root event/completion is inferred or supplied as privileged state.
        self.transitions = []
        h = (self.readout.reset(root.features[0], root.past_actions[0], task) if self.readout is not None
             else self.bc.encode_history(torch.stack(self.features)[None], torch.stack(self.past)[None])[0, -1])
        self.history = h.detach().clone()
        self.ready = True
        return self.history

    @torch.no_grad()
    def step(self, local_action):
        if not self.ready:
            raise RuntimeError('reset is required')
        self.ready = False  # Any exception invalidates the partially advanced state.
        tokens = self.world.action_spec.to_tokens(local_action.detach().cpu().numpy())
        _, frame = self.world.step(torch.as_tensor(tokens, device=self.world.device, dtype=torch.long)[None])
        pixels = self.frame_to_pixels(frame)
        if tuple(pixels.shape) != (1, 1, 3, 384, 640) or not torch.isfinite(pixels).all() or pixels.min() < -1 or pixels.max() > 1:
            raise ValueError('Generated observation violates P1 pixel contract')
        self.last_pixels = pixels.detach().clone()
        feature = self.encode_pixels(pixels).detach().clone()
        self.last_feature = feature
        past = torch.cat([F.one_hot(a, n) for a, n in zip(local_action, self.bc.action_sizes)]).to(feature.dtype)
        if self.readout is not None:
            nh, probability, block_boundary = self.readout.step(feature, past)
        else:
            block_boundary = len(self.features) == self.context
            if block_boundary:
                self.features, self.past = [], []
            self.features.append(feature); self.past.append(past)
            nh = self.bc.encode_history(torch.stack(self.features)[None], torch.stack(self.past)[None])[0, -1]
            self.transitions.append((self.history, nh))
            before = torch.stack([x[0] for x in self.transitions])[None]
            after = torch.stack([x[1] for x in self.transitions])[None]
            tasks = torch.full(before.shape[:2], self.task, device=before.device, dtype=torch.long)
            probability = float(self.reward(before, after, tasks)['completion'][0, -1].sigmoid())
        consistent = self.consistency(self.history, nh, local_action, self.task)
        if block_boundary:
            consistent = False  # P1 excluded these transitions from reward supervision.
        cut = False if self.risk is None else self.risk(self.history, nh, local_action, self.task)
        flags = {} if self.outcome is None else self.outcome(self.history, nh, local_action, self.task)
        if not isinstance(flags, dict) or set(flags) - {'terminated', 'timeout', 'uncertainty'}:
            self.ready = False
            raise ValueError('Outcome callback must return explicit supported ModelStep fields')
        if type(cut) is not bool or type(consistent) is not bool:
            raise ValueError('Risk and consistency callbacks must return booleans')
        if any(type(flags[k]) is not bool for k in ('terminated','timeout') if k in flags):
            raise ValueError('Outcome terminal/timeout flags must be boolean')
        if 'uncertainty' in flags:
            import math
            if not math.isfinite(flags['uncertainty']) or not 0 <= flags['uncertainty'] <= 1:
                raise ValueError('Invalid measured uncertainty')
        self.history = nh.detach().clone()
        self.ready = not (cut or flags.get('terminated',False) or flags.get('timeout',False))
        return ModelStep(self.history, probability, consistent, uncertainty_cut=cut, **flags)


    @torch.no_grad()
    def begin_skill(self, task):
        """Change a task-origin goal without resetting or re-encoding the world state.

        Reward/policy memory starts at this current frame, matching the bound
        task-origin training protocol; the episode ledger lives in SkillExecutor.
        """
        if not self.ready or self.contract is None:
            raise ValueError('A valid contracted backend state is required for handoff')
        self.contract.reward_task(task)
        self.ready = False
        past = torch.zeros(sum(self.bc.action_sizes),device=self.last_feature.device,dtype=self.last_feature.dtype)
        self.history = self.readout.reset(self.last_feature,past,task)
        self.task = task
        self.ready = True
        return self.history.detach().clone()


    @torch.no_grad()
    def decision_history(self):
        """Canonical next high-level state after a successful goal, without mutating WM."""
        if not self.ready or self.contract is None:
            raise ValueError('A valid contracted backend state is required')
        past = torch.zeros(sum(self.bc.action_sizes),device=self.last_feature.device,dtype=self.last_feature.dtype)
        history, _ = self.bc.encode_step(self.last_feature[None],past[None])
        return history[0].detach().clone()
