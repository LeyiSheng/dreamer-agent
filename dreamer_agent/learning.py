"""Explicit time/boundary semantics and independent BC/PMPO objectives."""
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, kl_divergence


def lambda_returns(rewards: Tensor, next_values: Tensor, discounts: Tensor,
                   terminated: Tensor, boundary: Tensor, lam: float = .95) -> Tensor:
    """[time,batch]; boundary stops recursion, not bootstrap.

    rewards/discounts can be micro or duration-aggregated macro quantities.
    Invalid uncertainty transitions must be removed before calling; supply a
    bounded value at the last valid prefix state via next_values.
    """
    if rewards.ndim != 2 or rewards.numel() == 0:
        raise ValueError('Expected nonempty [time,batch] tensors')
    if any(x.shape != rewards.shape for x in (next_values, discounts, terminated, boundary)):
        raise ValueError('Return tensors must have identical shapes')
    if terminated.dtype != torch.bool or boundary.dtype != torch.bool:
        raise ValueError('Termination and boundary masks must be boolean')
    if not 0 <= lam <= 1 or not torch.all((discounts >= 0) & (discounts <= 1)):
        raise ValueError('Invalid discount or lambda')
    if not all(torch.isfinite(x).all() for x in (rewards, next_values, discounts)):
        raise ValueError('Nonfinite return inputs')
    with torch.no_grad():
        out = torch.empty_like(rewards)
        carry = next_values[-1]
        for t in reversed(range(len(rewards))):
            tail = torch.where(boundary[t], next_values[t], carry)
            continuation = (1 - lam) * next_values[t] + lam * tail
            out[t] = rewards[t] + discounts[t] * (~terminated[t]) * continuation
            carry = out[t]
        return out


def macro_reward(rewards: Tensor, gamma: float) -> tuple[Tensor, float]:
    if rewards.ndim != 1 or not len(rewards) or not 0 <= gamma <= 1:
        raise ValueError('Expected nonempty skill rewards and valid gamma')
    weights = rewards.new_tensor(gamma) ** torch.arange(len(rewards), device=rewards.device)
    return (rewards * weights).sum(), gamma ** len(rewards)


class HistoryPolicy(nn.Module):
    """Causal history + task MTP. Inputs are frozen visual features and past actions.

    Each call contains one contiguous episode segment per batch item. Heads use
    local categorical IDs; a separately verified codec maps them to WM actions.
    Deployment executes prediction distance zero and observes again.
    """
    def __init__(self, feature_dim: int, past_action_dim: int, tasks: int,
                 action_sizes: tuple[int, ...], hidden: int = 128, targets: int = 8, task_conditioned: bool = True):
        super().__init__()
        if not action_sizes or min(action_sizes) < 2 or targets < 1:
            raise ValueError('Only nonconstant action factors are learned')
        self.targets, self.action_sizes = targets, action_sizes
        self.history = nn.GRU(feature_dim + past_action_dim, hidden, batch_first=True)
        self.task_conditioned = task_conditioned
        self.task = nn.Embedding(tasks, hidden) if task_conditioned else None
        self.heads = nn.ModuleList(nn.Linear(hidden, targets * n) for n in action_sizes)

    def encode_history(self, features: Tensor, past_actions: Tensor):
        """Task-independent causal state shared with frozen reward readout."""
        h, _ = self.history(torch.cat((features, past_actions), dim=-1))
        return h

    @torch.no_grad()
    def encode_step(self, features: Tensor, past_actions: Tensor, state=None):
        """Frozen one-frame inference; caller resets state at the trained block boundary."""
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Incremental history requires a frozen eval policy')
        if features.ndim != 2 or past_actions.ndim != 2 or features.shape[0] != past_actions.shape[0]:
            raise ValueError('Expected one feature and past action per batch item')
        x = torch.cat((features, past_actions), -1)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite history input')
        h, state = self.history(x[:, None], state)
        return h[:, 0].detach(), state.detach()

    def forward(self, features: Tensor, past_actions: Tensor, tasks: Tensor):
        h = self.encode_history(features, past_actions)
        if self.task is not None:
            h = h + self.task(tasks)
        return [head(h).reshape(*h.shape[:2], self.targets, n)
                for head, n in zip(self.heads, self.action_sizes)]


def bc_loss(logits: list[Tensor], targets: Tensor, valid: Tensor, *, normalization="valid_mean") -> Tensor:
    """Targets [batch,time,MTP,factor]; valid [batch,time,MTP].

    valid_mean preserves existing runs; anchor_sum implements V5's sum over
    prediction distances averaged over time anchors with any valid label.
    The choice must be bound with the BC mixture coefficient in RLConfig.
    """
    if normalization not in {'valid_mean', 'anchor_sum'}:
        raise ValueError('Unknown BC normalization')
    if not logits or targets.shape != (*valid.shape, len(logits)) or valid.dtype != torch.bool:
        raise ValueError('Invalid BC shapes/mask')
    if not valid.any():
        raise ValueError('No valid supervision')
    loss = 0
    for i, scores in enumerate(logits):
        if scores.shape[:-1] != valid.shape:
            raise ValueError('Logits and mask mismatch')
        loss = loss - Categorical(logits=scores[valid]).log_prob(targets[..., i][valid]).mean()
    if normalization == 'anchor_sum':
        loss = loss * (valid.sum() / valid.any(-1).sum())
    return loss


def pmpo_loss(logits: list[Tensor], prior_logits: list[Tensor], actions: Tensor,
              advantages: Tensor, alpha: float = .5, beta: float = .3) -> Tensor:
    """Fresh imagined actions; sum exact categorical log-prob and KL over factors.

    Prior and advantage are detached. This function does not authorize RL before
    P0/P1 gates, and does not implement stale-replay importance correction.
    """
    if not 0 <= alpha <= 1 or beta < 0 or not logits or len(logits) != len(prior_logits):
        raise ValueError('Invalid PMPO configuration')
    if advantages.numel() == 0 or actions.shape != (*advantages.shape, len(logits)):
        raise ValueError('Invalid action/advantage shape')
    if not torch.isfinite(advantages).all():
        raise ValueError('Nonfinite advantages')
    logp, kl = 0, 0
    for i, (scores, prior) in enumerate(zip(logits, prior_logits)):
        if scores.shape != prior.shape or scores.shape[:-1] != advantages.shape:
            raise ValueError('Policy/prior shape mismatch')
        pi, ref = Categorical(logits=scores), Categorical(logits=prior.detach())
        logp = logp + pi.log_prob(actions[..., i])
        kl = kl + kl_divergence(pi, ref)
    positive = advantages.detach() >= 0
    loss = beta * kl.mean()
    if positive.any():
        loss = loss - alpha * logp[positive].mean()
    if (~positive).any():
        loss = loss + (1-alpha) * logp[~positive].mean()
    return loss
