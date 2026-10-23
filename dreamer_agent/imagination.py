"""P2 single-stream imagination engine. No environment or privileged-state inputs.

A backend owns frozen WM/history/reward state. Its reset/step return frozen BC
history and calibrated reward probabilities; it must implement the separately
reviewed pixel/action alignment contract. Synthetic backends are tests only.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .learning import lambda_returns, pmpo_loss, bc_loss
from .contracts import Episode, MilestoneLedger
from .rl_data import OfflineBatch


@dataclass(frozen=True)
class RLConfig:
    horizon: int = 16
    gamma: float = .997
    lam: float = .95
    alpha: float = .5
    beta: float = .3
    bc_weight: float = 1.
    learning_rate: float = 3e-4
    confirmation_steps: int = 2
    bootstrap_min: float = 0.
    bootstrap_max: float = 1.
    bc_normalization: str = "valid_mean"

    def __post_init__(self):
        if self.bc_normalization not in {"valid_mean", "anchor_sum"}:
            raise ValueError("Unknown BC normalization")
        if self.horizon not in (16, 32, 64):
            raise ValueError('Only reviewed 16/32/64 horizons are supported')
        values = (self.gamma, self.lam, self.alpha, self.beta, self.bc_weight,
                  self.learning_rate, self.bootstrap_min, self.bootstrap_max)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite RL configuration')
        if not all(0 <= x <= 1 for x in (self.gamma, self.lam, self.alpha)):
            raise ValueError('Invalid discount/lambda/group weight')
        if min(self.beta, self.bc_weight) < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid loss/optimizer weight')
        if type(self.confirmation_steps) is not int or self.confirmation_steps < 1:
            raise ValueError('Positive confirmation length required')
        if self.bootstrap_min > self.bootstrap_max:
            raise ValueError('Invalid bounded bootstrap range')


class PolicyReadout(nn.Module):
    """Initialize all MTP heads from BC, while keeping BC history immutable."""
    def __init__(self, bc):
        super().__init__()
        self.task = deepcopy(bc.task)
        self.heads = deepcopy(bc.heads)
        self.targets, self.action_sizes = bc.targets, bc.action_sizes

    def forward(self, history, tasks, *, all_targets=False):
        h = history.detach()
        if self.task is not None:
            h = h + self.task(tasks)
        logits = [head(h).reshape(*h.shape[:-1], self.targets, size)
                  for head, size in zip(self.heads, self.action_sizes)]
        return logits if all_targets else [x[..., 0, :] for x in logits]


class ValueHead(nn.Module):
    def __init__(self, history_dim, tasks):
        super().__init__()
        self.task = nn.Embedding(tasks, history_dim)
        self.net = nn.Sequential(nn.Linear(history_dim + 1, history_dim), nn.SiLU(), nn.Linear(history_dim, 1))

    def forward(self, history, tasks, confirmation):
        x = torch.cat((history.detach() + self.task(tasks), confirmation.detach().unsqueeze(-1)), -1)
        return self.net(x).squeeze(-1)


@dataclass
class ModelStep:
    history: torch.Tensor
    completion_probability: float
    consistent: bool
    uncertainty_cut: bool = False
    terminated: bool = False
    timeout: bool = False
    uncertainty: float = 0.


@dataclass
class Trajectory:
    history: torch.Tensor
    next_history: torch.Tensor
    actions: torch.Tensor
    tasks: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    boundary: torch.Tensor
    bounded_bootstrap: torch.Tensor
    policy_version: int
    source_episode_id: str
    model_calls: int
    uncertainty_cuts: int
    confirmation: torch.Tensor
    next_confirmation: torch.Tensor
    rewarded: torch.Tensor
    seed: int
    uncertainties: torch.Tensor
    used: bool = False


class ImaginationTrainer:
    """Fresh single-episode rollouts only; no stale replay PMPO updates.

    backend.reset(root, task, seed) -> h[H]
    backend.step(local_action[K]) -> ModelStep
    backend.frozen_modules -> tuple of all frozen torch modules
    root.episode must be an offline TRAIN Episode; reset verifies root payload.
    """
    def __init__(self, bc, backend, config, thresholds, *, seed=0, data_binding=None, policy_readout=None, history_dim=None, tasks=None):
        self.config, self.backend = config, backend
        self.actor = (PolicyReadout(bc) if policy_readout is None else deepcopy(policy_readout)).requires_grad_(True)
        self.prior = deepcopy(self.actor).eval().requires_grad_(False)
        tasks = tasks if tasks is not None else (bc.task.num_embeddings if bc.task is not None else 1)
        history_dim = history_dim if history_dim is not None else bc.history.hidden_size
        # CPU initialization inside fork_rng does not perturb caller CPU/CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.value = ValueHead(history_dim, tasks)
        self.value = self.value.to(next(bc.parameters()).device)
        self.seed, self.data_binding = seed, data_binding
        self.thresholds = dict(thresholds)
        if not self.thresholds or any(type(k) is not int or not 0 <= k < tasks or
                not math.isfinite(v) or not 0 <= v <= 1 for k, v in self.thresholds.items()):
            raise ValueError('Explicit calibrated task thresholds required')
        self.frozen = (bc, self.prior, *backend.frozen_modules)
        self._check_frozen()
        self.optimizer = torch.optim.AdamW([*self.actor.parameters(), *self.value.parameters()],
                                           lr=config.learning_rate)
        self.generator = torch.Generator(device=next(bc.parameters()).device).manual_seed(seed)
        self.version = 0

    def _check_frozen(self):
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in self.frozen):
            raise ValueError('WM, reward, history, and prior must be frozen and eval')

    @staticmethod
    def _history(h):
        if h.ndim != 1 or not torch.isfinite(h).all():
            raise ValueError('Expected finite single-stream history vector')
        return h.detach().clone()

    @torch.no_grad()
    def collect(self, root, task, *, seed):
        self._check_frozen()
        ep = root.episode
        if not isinstance(ep, Episode) or ep.origin != 'offline' or ep.split != 'train' or not ep.source_episode_id:
            raise ValueError('Imagination roots must be attributable offline training contexts')
        if task not in self.thresholds:
            raise ValueError('Task has no approved completion threshold')
        h = self._history(self.backend.reset(root, task, seed))
        tid = torch.tensor(task, device=h.device, dtype=torch.long)
        ledger, streak, rows = MilestoneLedger(), 0, []
        calls, cuts = 0, 0
        for _ in range(self.config.horizon):
            logits = self.actor(h, tid)
            action = torch.stack([torch.multinomial(x.softmax(-1), 1, generator=self.generator)[0] for x in logits])
            step = self.backend.step(action)
            calls += 1
            if type(step.uncertainty_cut) is not bool or type(step.terminated) is not bool or type(step.timeout) is not bool:
                raise ValueError('Explicit boolean boundary semantics required')
            if step.uncertainty_cut:
                cuts += 1
                if rows:
                    rows[-1][5] = True
                    rows[-1][6] = True
                break  # Discard this entire transition, including reward and successor.
            nh = self._history(step.history)
            p = float(step.completion_probability)
            if not math.isfinite(p) or not 0 <= p <= 1 or type(step.consistent) is not bool:
                raise ValueError('Invalid reward readout')
            before_streak = streak / self.config.confirmation_steps
            if not math.isfinite(step.uncertainty) or not 0 <= step.uncertainty <= 1:
                raise ValueError("Invalid uncertainty proxy")
            streak = streak + 1 if p >= self.thresholds[task] and step.consistent else 0
            reward = ledger.observe(str(task), confirmed=streak >= self.config.confirmation_steps)
            # Completion does not terminate the environment or reset the ledger.
            rows.append([h, nh, action, reward, step.terminated, step.timeout or step.terminated or bool(reward), False,
                         before_streak, streak / self.config.confirmation_steps, step.uncertainty])
            h = nh
            if step.terminated or step.timeout or reward:
                break  # Objective complete: zero remaining objective return, NOT environment termination.
        if not rows:
            return None, dict(model_calls=calls, uncertainty_cuts=cuts, valid_steps=0)
        rows[-1][5] = True  # Horizon/timeout: bootstrap, never recurse into another root.
        def tensor(column, dtype):
            return torch.tensor([r[column] for r in rows], device=h.device, dtype=dtype)
        trajectory = Trajectory(torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows]),
            torch.stack([r[2] for r in rows]), tid.expand(len(rows)), tensor(3, h.dtype),
            tensor(4, torch.bool), tensor(5, torch.bool), tensor(6, torch.bool),
            self.version, ep.source_episode_id, calls, cuts, tensor(7, h.dtype), tensor(8, h.dtype),
            tensor(3, h.dtype).bool(), seed, tensor(9, h.dtype))
        return trajectory, dict(model_calls=calls, uncertainty_cuts=cuts, valid_steps=len(rows))

    def update(self, trajectory, offline):
        """offline=(frozen history[B,T,H], tasks[B,T], targets[B,T,M,K], mask[B,T,M])."""
        self._check_frozen()
        t, c = trajectory, self.config
        if t.used or t.policy_version != self.version:
            raise ValueError('PMPO requires a fresh, unused rollout from the current policy')
        with torch.no_grad():
            values = self.value(t.history, t.tasks, t.confirmation)
            nv = self.value(t.next_history, t.tasks, t.next_confirmation)
            nv = torch.where(t.bounded_bootstrap, nv.clamp(c.bootstrap_min, c.bootstrap_max), nv)
            nv = torch.where(t.rewarded, torch.zeros_like(nv), nv)
            returns = lambda_returns(t.rewards[:, None], nv[:, None],
                torch.full_like(nv[:, None], c.gamma), t.terminated[:, None], t.boundary[:, None], c.lam)[:, 0]
            advantages = returns - values
            prior = self.prior(t.history, t.tasks)
        actor_loss = pmpo_loss(self.actor(t.history, t.tasks), prior, t.actions, advantages, c.alpha, c.beta)
        value_loss = F.mse_loss(self.value(t.history, t.tasks, t.confirmation), returns)
        bc = actor_loss.new_zeros(())
        if c.bc_weight:
            if offline is None:
                raise ValueError('Offline BC batch required for mixed optimization')
            if not isinstance(offline, OfflineBatch):
                raise ValueError("Provenance-bearing OfflineBatch required")
            h, tasks, targets, mask = offline.validate(self.data_binding)
            bc = bc_loss(self.actor(h, tasks, all_targets=True), targets, mask, normalization=c.bc_normalization)
        loss = actor_loss + value_loss + c.bc_weight * bc
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite RL loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.value.parameters()], 1., error_if_nonfinite=True)
        self.optimizer.step()
        t.used = True
        self.version += 1
        return dict(update=self.version, actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
            bc_loss=float(bc.detach()), bc_normalization=c.bc_normalization, imagined_reward=float(t.rewards.sum()), valid_steps=len(t.rewards),
            model_calls=t.model_calls, uncertainty_cuts=t.uncertainty_cuts, reward_source='learned_imagination',
            source_episode_id=t.source_episode_id, rollout_seed=t.seed, training_seed=self.seed,
            task=int(t.tasks[0]), uncertainty_mean=float(t.uncertainties.mean()),
            terminated=bool(t.terminated[-1]), objective_complete=bool(t.rewarded[-1]))
