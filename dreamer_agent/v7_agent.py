"""V7 pixel-only agent: VPT-3x backbone -> belief/event heads -> scheduler -> skill policy.

The inferred ledger L-hat (completion latches seen so far) and belief counts come only
from pixels; true inventory is never read. Used by eval_long (B0/B1 arms) and by the
imagination trainer.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .v7_data import BUCKETS, ITEMS, MILESTONES, SKILLS
from .vpt_heads import SignalHeads, SkillPolicy


def believed_counts(belief, confident=0.5):
    """belief [I, Bk] probs -> dict item -> lower-edge count, or None if the mode is not confident."""
    out = {}
    for i, item in enumerate(ITEMS):
        p = belief[i]
        j = int(p.argmax())
        out[item] = BUCKETS[j] if p[j] >= confident else None
    return out


# (skill, done-predicate over believed counts c and ledger L); first unmet entry is chosen.
TECH_TREE = [
    ('log', lambda c, L: (c['log'] or 0) >= 4 or (c['planks'] or 0) >= 8 or 'wooden_pickaxe' in L),
    ('planks', lambda c, L: (c['planks'] or 0) >= 8 or 'wooden_pickaxe' in L),
    ('crafting_table', lambda c, L: 'crafting_table' in L),
    ('stick', lambda c, L: (c['stick'] or 0) >= 2 or 'wooden_pickaxe' in L),
    ('wooden_pickaxe', lambda c, L: 'wooden_pickaxe' in L),
    ('cobblestone', lambda c, L: (c['cobblestone'] or 0) >= 11 or 'stone_pickaxe' in L and 'furnace' in L),
    ('stone_pickaxe', lambda c, L: 'stone_pickaxe' in L),
    ('furnace', lambda c, L: 'furnace' in L),
    ('iron_ore', lambda c, L: (c['iron_ore'] or 0) >= 3 or 'iron_ingot' in L and (c['iron_ingot'] or 0) >= 3 or 'iron_pickaxe' in L),
    ('iron_ingot', lambda c, L: (c['iron_ingot'] or 0) >= 3 or 'iron_pickaxe' in L),
    ('iron_pickaxe', lambda c, L: 'iron_pickaxe' in L),
    ('diamond', lambda c, L: False),
]


@dataclass
class SchedulerState:
    skill: int = 0
    steps_in_skill: int = 0
    general_left: int = 0
    ledger: set = field(default_factory=set)
    latch_run: dict = field(default_factory=dict)


class HandScheduler:
    """B0 hand-written tech-tree scheduler driven by pixel beliefs (PLAN_V7 §8 Q3).

    A milestone enters the ledger once its latch P(count>=1) exceeds `latch_threshold`
    for `hold` consecutive steps. Skills time out after `max_steps` and fall back to the
    general mode for `general_steps` (exploration) before retrying.
    """

    def __init__(self, latch_threshold=0.7, hold=10, max_steps=2400, general_steps=400):
        self.latch_threshold, self.hold, self.max_steps, self.general_steps = latch_threshold, hold, max_steps, general_steps

    def reset(self):
        return SchedulerState()

    def step(self, state, signals):
        latch = signals['latch_mean']
        for m in MILESTONES:
            run = state.latch_run.get(m, 0) + 1 if latch[ITEMS.index(m)] > self.latch_threshold else 0
            state.latch_run[m] = run
            if run >= self.hold:
                state.ledger.add(m)
        counts = believed_counts(signals['belief'])
        target = next(SKILLS.index(s) for s, done in TECH_TREE if not done(counts, state.ledger))
        state.steps_in_skill += 1
        if state.general_left > 0:
            state.general_left -= 1
            return 0
        if target != state.skill:
            state.skill, state.steps_in_skill = target, 0
        elif state.steps_in_skill > self.max_steps:
            state.skill, state.steps_in_skill, state.general_left = 0, 0, self.general_steps
        return state.skill


class V7Agent:
    def __init__(self, vpt, heads_path, policy_path, device, scheduler=None):
        self.vpt, self.device = vpt, device
        blob = torch.load(heads_path, map_location='cpu', weights_only=False)
        self.heads = SignalHeads(blob['members']).to(device).eval()
        self.heads.load_state_dict(blob['state'])
        self.policy = SkillPolicy(vpt.pi_head).to(device).eval()
        if policy_path:
            sp = torch.load(policy_path, map_location='cpu', weights_only=False)
            self.policy.pi_head.load_state_dict(sp['pi_head'])
            self.policy.gamma.load_state_dict(sp['gamma'])
            self.policy.beta.load_state_dict(sp['beta'])
        self.scheduler = scheduler or HandScheduler()

    def reset(self):
        self.state = self.vpt.initial_state(1)
        self.h_prev = None
        self.sched = self.scheduler.reset()
        self.first = True

    @torch.no_grad()
    def act(self, image, generator):
        """image: 128x128x3 uint8 (vpt_features.to_vpt_input). Returns (buttons, camera, skill, signals)."""
        x = torch.from_numpy(image.copy())[None, None].to(self.device)
        (h, _), self.state = self.vpt.net({'img': x}, self.state, context={'first': torch.tensor([[self.first]], device=self.device)})
        self.first = False
        h = h[0, 0].float()
        prev = h if self.h_prev is None else self.h_prev
        self.h_prev = h
        sig = self.heads.signals(h[None], prev[None])
        signals = {k: v[0].cpu().numpy() for k, v in sig.items()}
        skill = self.scheduler.step(self.sched, signals)
        logp = self.policy(h[None], torch.tensor([skill], device=self.device))
        buttons = torch.multinomial(logp['buttons'].exp(), 1, generator=generator)
        camera = torch.multinomial(logp['camera'].exp(), 1, generator=generator)
        return int(buttons), int(camera), skill, signals
