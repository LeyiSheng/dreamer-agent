"""V7 heads on the frozen VPT-3x feature h_t (PLAN_V7 §3, §4.2, §7.1).

- SignalHeads: event head e_k(h_{t-1}, h_t) (transient) and inventory belief b_t
  (bucketed counts per item). The completion latch c_k = P(count_k >= 1) is read
  from the belief, so latch and belief cannot disagree.
- SkillPolicy: original pi_head applied to a skill-conditioned feature
  h * (1 + gamma_k) + beta_k with zero-initialised gamma/beta, so every skill starts
  exactly at bc-early-game-3x; skill 0 (general) is kept at the original head.
- TwoHotValue: symexp twohot value, 255 bins on symlog [-20, 20].
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .v7_data import BUCKETS, ITEMS, MILESTONES, SKILLS

DIM = 3072


def mlp(inp, hidden, out, layers=2):
    mods, width = [], inp
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        width = hidden
    mods.append(nn.Linear(width, out))
    return nn.Sequential(*mods)


class SignalHead(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.event = mlp(2 * DIM, hidden, len(MILESTONES))
        self.belief = mlp(DIM, hidden, len(ITEMS) * len(BUCKETS))

    def forward(self, h, h_prev):
        event = self.event(torch.cat([h, h - h_prev], -1))
        belief = self.belief(h).unflatten(-1, (len(ITEMS), len(BUCKETS)))
        return event, belief


class SignalHeads(nn.Module):
    """Ensemble of M independently initialised heads (shared frozen backbone; not independent experts)."""

    def __init__(self, members=3, hidden=512):
        super().__init__()
        self.members = nn.ModuleList(SignalHead(hidden) for _ in range(members))

    def forward(self, h, h_prev):
        outs = [m(h, h_prev) for m in self.members]
        return torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs])  # [M,B,K], [M,B,I,Bk]

    @torch.no_grad()
    def signals(self, h, h_prev):
        event, belief = self(h, h_prev)
        p_event = event.sigmoid()
        p_belief = belief.softmax(-1)
        latch = 1 - p_belief[..., 0]  # P(count >= 1)
        edges = torch.tensor(BUCKETS, dtype=p_belief.dtype, device=p_belief.device)
        return dict(event_mean=p_event.mean(0), event_std=p_event.std(0), latch_mean=latch.mean(0), latch_std=latch.std(0),
                    belief=p_belief.mean(0), count_lower=(p_belief.mean(0) * edges).sum(-1))


class SkillPolicy(nn.Module):
    def __init__(self, pi_head):
        super().__init__()
        self.pi_head = copy.deepcopy(pi_head)
        self.gamma = nn.Embedding(len(SKILLS), DIM)
        self.beta = nn.Embedding(len(SKILLS), DIM)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def condition(self, h, skill):
        general = (skill == 0).unsqueeze(-1)
        conditioned = h * (1 + self.gamma(skill)) + self.beta(skill)
        return torch.where(general, h, conditioned)

    def forward(self, h, skill):
        """Returns log-probs {'buttons': [..., 8641], 'camera': [..., 121]} for h [..., DIM]."""
        x = self.condition(h, skill)
        shape = x.shape[:-1]
        pd = self.pi_head(x.reshape(-1, 1, DIM))
        return {k: pd[k].reshape(*shape, pd[k].shape[-1]).float().log_softmax(-1) for k in ('buttons', 'camera')}


_CAMERA_ON = {}


def camera_on(device):
    """bool [8641]: joint button actions whose camera meta action is on. When it is off the
    camera head is ignored by the env mapping (VPT CameraHierarchicalMapping) and gives ~0
    probability to the null bin, so its log-prob must not enter likelihoods or KLs."""
    key = str(device)
    if key not in _CAMERA_ON:
        from .vpt_features import _import_vpt
        _import_vpt()
        from lib.action_mapping import CameraHierarchicalMapping
        off = CameraHierarchicalMapping(n_camera_bins=11).BUTTON_IDX_TO_CAMERA_META_OFF
        _CAMERA_ON[key] = torch.from_numpy(~off).to(device)
    return _CAMERA_ON[key]


def action_logprob(logp, buttons, camera):
    on = camera_on(buttons.device)[buttons]
    lp_camera = logp['camera'].gather(-1, camera.unsqueeze(-1)).squeeze(-1)
    return logp['buttons'].gather(-1, buttons.unsqueeze(-1)).squeeze(-1) + torch.where(on, lp_camera, torch.zeros_like(lp_camera))


def action_kl(logp, prior):
    """Exact KL of the hierarchical action distribution: KL_buttons + P(camera on) * KL_camera."""
    p_buttons = logp['buttons'].exp()
    kl_buttons = (p_buttons * (logp['buttons'] - prior['buttons'])).sum(-1)
    kl_camera = (logp['camera'].exp() * (logp['camera'] - prior['camera'])).sum(-1)
    p_on = (p_buttons * camera_on(p_buttons.device)).sum(-1)
    return kl_buttons + p_on * kl_camera


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHotValue(nn.Module):
    def __init__(self, inp=DIM + len(SKILLS) + 2, hidden=512, bins=255, low=-20.0, high=20.0):
        super().__init__()
        self.net = mlp(inp, hidden, bins)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('support', torch.linspace(low, high, bins))

    def logits(self, x):
        return self.net(x)

    def value(self, x):
        return symexp((self.logits(x).softmax(-1) * self.support).sum(-1))

    def loss(self, x, target):
        y = symlog(target).clamp(self.support[0], self.support[-1])
        idx = torch.searchsorted(self.support, y.contiguous()).clamp(1, len(self.support) - 1)
        lo, hi = self.support[idx - 1], self.support[idx]
        w_hi = ((y - lo) / (hi - lo)).clamp(0, 1)
        dist = torch.zeros(*y.shape, len(self.support), device=y.device)
        dist.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        dist.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        return -(dist * self.logits(x).log_softmax(-1)).sum(-1)


def value_input(h, skill, progress, remaining):
    """Augmented state for values: h, one-hot skill, in-skill progress p and remaining time tau (both in [0,1])."""
    return torch.cat([h, F.one_hot(skill, len(SKILLS)).float(), progress.unsqueeze(-1), remaining.unsqueeze(-1)], -1)
