"""P1 supervised event/completion heads on frozen BC history.

Labels are external supervision, never model inputs. Unknown labels need false
masks. Probabilities alone are not calibrated rewards and never authorize RL.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RewardCompletionHeads(nn.Module):
    def __init__(self, history_dim, tasks, hidden=128):
        super().__init__()
        if min(history_dim, tasks, hidden) < 1:
            raise ValueError("Dimensions must be positive")
        self.history_dim = history_dim
        self.task = nn.Embedding(tasks, hidden)
        self.transition = nn.Linear(2 * history_dim, hidden)
        self.event = nn.Linear(hidden, 1)
        self.completion = nn.Linear(hidden, 1)

    def forward(self, history, next_history, tasks):
        if history.shape != next_history.shape or history.shape[-1] != self.history_dim:
            raise ValueError("History shape mismatch")
        if tasks.shape != history.shape[:-1] or tasks.dtype != torch.long:
            raise ValueError("Expected one integer task per transition")
        if not torch.isfinite(history).all() or not torch.isfinite(next_history).all():
            raise ValueError("Nonfinite history")
        if ((tasks < 0) | (tasks >= self.task.num_embeddings)).any():
            raise ValueError("Unknown task")
        # Hard isolation even if caller accidentally passes an attached BC graph.
        x = torch.cat((history.detach(), next_history.detach()), dim=-1)
        h = F.silu(self.transition(x) + self.task(tasks))
        return dict(event=self.event(h).squeeze(-1),
                    completion=self.completion(h).squeeze(-1))


def supervised_head_loss(logits, labels, masks):
    """Independent binary labels; absent supervision contributes no gradient.

    Event denotes a transition event; completion denotes skill completion.
    Neither is environment termination. Class weighting/sampling changes require
    independent probability calibration and are not enabled by this objective.
    """
    names = {"event", "completion"}
    if set(logits) != names or set(labels) != names or set(masks) != names:
        raise ValueError("Both head fields are required")
    losses = []
    for name in sorted(names):
        score, target, valid = logits[name], labels[name], masks[name]
        if score.shape != target.shape or score.shape != valid.shape or valid.dtype != torch.bool:
            raise ValueError("Invalid supervision shape/mask")
        if valid.any():
            y = target[valid]
            if not torch.isfinite(score[valid]).all() or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
                raise ValueError("Supervised labels must be binary and finite")
            losses.append(F.binary_cross_entropy_with_logits(score[valid], y.to(score.dtype)))
    if not losses:
        raise ValueError("No supervised labels")
    return torch.stack(losses).mean()


@torch.no_grad()
def frozen_history(policy, features, past_actions):
    """Read task-independent history from an explicitly frozen, eval BC model."""
    if policy.training or any(p.requires_grad for p in policy.parameters()):
        raise ValueError("Freeze the BC model and set eval before reward training")
    return policy.encode_history(features, past_actions).detach()


class TemporalRewardCompletionHeads(nn.Module):
    """Causal reward memory spanning the full task clip; resets per call/episode.

    Frozen BC readouts may reset every 32 frames. This separate recurrent state
    retains observations across those boundaries, without any true event input.
    Right padding must be masked by the caller. No future observations are used.
    """
    def __init__(self, history_dim, tasks, hidden=128):
        super().__init__()
        self.task = nn.Embedding(tasks, hidden)
        self.transition = nn.Linear(2 * history_dim, hidden)
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.event = nn.Linear(hidden, 1)
        self.completion = nn.Linear(hidden, 1)

    def forward(self, history, next_history, tasks):
        if history.ndim != 3 or history.shape != next_history.shape or tasks.shape != history.shape[:2]:
            raise ValueError("Expected batch,time,history and matching tasks")
        x = F.silu(self.transition(torch.cat((history.detach(), next_history.detach()), -1)) + self.task(tasks))
        h, _ = self.memory(x)
        return dict(event=self.event(h).squeeze(-1), completion=self.completion(h).squeeze(-1))


    @torch.no_grad()
    def forward_step(self, history, next_history, tasks, state=None):
        """One transition, preserving task-origin reward memory across BC resets."""
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Incremental reward requires a frozen eval model')
        if history.ndim != 2 or history.shape != next_history.shape or tasks.shape != history.shape[:1]:
            raise ValueError('Expected batch,history and one task per batch')
        if not torch.isfinite(history).all() or not torch.isfinite(next_history).all():
            raise ValueError('Nonfinite reward history')
        x = F.silu(self.transition(torch.cat((history.detach(), next_history.detach()), -1)) + self.task(tasks))
        h, state = self.memory(x[:, None], state)
        h = h[:, 0]
        return dict(event=self.event(h).squeeze(-1), completion=self.completion(h).squeeze(-1)), state.detach()
