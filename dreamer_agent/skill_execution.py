"""Budgeted fixed-skill execution and SMDP accounting; no privileged inventory.

This is P2 fixed scheduling / P3 interface preparation, not a learned high-level
policy, a reward calibration, or an authorization to run real-model training.
"""
from dataclasses import dataclass
import math
import torch
from .contracts import MilestoneLedger


@dataclass(frozen=True)
class MacroTransition:
    skill: str
    history: torch.Tensor
    next_history: torch.Tensor
    reward: float
    discount: float
    duration: int
    reason: str
    terminated: bool
    completed: bool
    achieved_before: tuple = ()
    achieved_after: tuple = ()

    def target(self, next_value, *, bootstrap_bounds=None):
        """Skill completion/timeout bootstrap; only environment termination cancels it."""
        value = torch.as_tensor(next_value).detach()
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite macro bootstrap')
        if self.terminated:
            value = torch.zeros_like(value)
        elif self.reason == 'uncertainty_cut':
            if bootstrap_bounds is None or len(bootstrap_bounds) != 2:
                raise ValueError('Uncertainty prefix requires explicit global-return bounds')
            low, high = bootstrap_bounds
            if not all(math.isfinite(x) for x in (low, high)) or low > high:
                raise ValueError('Invalid macro bootstrap bounds')
            value = value.clamp(low, high)
        return self.reward + self.discount * value


class SkillExecutor:
    """One instance per episode; beginning another skill never clears its ledger."""
    def __init__(self, episode_id, *, gamma=.997, confirmation_steps=2):
        if not episode_id or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError('Explicit episode and valid discount required')
        if type(confirmation_steps) is not int or confirmation_steps < 1:
            raise ValueError('Positive confirmation length required')
        self.episode_id, self.gamma, self.confirmation_steps = episode_id, gamma, confirmation_steps
        self.ledger = MilestoneLedger()
        self.active = False
        self.ended = False

    @staticmethod
    def valid_history(history):
        if history.ndim != 1 or not torch.isfinite(history).all():
            raise ValueError('Finite history vector required')
        return history.detach().clone()

    def begin(self, skill, history, *, threshold, max_steps):
        if self.active or self.ended:
            raise ValueError('Finish current skill; a terminated/cut episode cannot be reused')
        if not isinstance(skill,str) or not skill or skill == 'general_behavior':
            raise ValueError('An explicit reward-bearing skill is required')
        if not math.isfinite(threshold) or not 0 <= threshold <= 1 or type(max_steps) is not int or max_steps < 1:
            raise ValueError('Valid threshold and skill budget required')
        self.achieved_before = tuple(sorted(self.ledger.achieved))
        self.history = self.valid_history(history)
        self.next_history = self.history.clone()
        self.skill, self.threshold, self.max_steps = skill,threshold,max_steps
        self.duration, self.streak, self.reward = 0,0,0.
        self.active = True

    def advance(self, step):
        if not self.active:
            raise RuntimeError('Begin a skill before consuming transitions')
        if any(type(x) is not bool for x in (step.terminated,step.timeout,step.uncertainty_cut)):
            raise ValueError('Boolean boundary flags required')
        if step.uncertainty_cut:
            # Do not consume the invalid successor/reward, even if it looks successful.
            self.ended = True
            return self._finish('uncertainty_cut',False,False)
        history = self.valid_history(step.history)
        if history.shape != self.history.shape:
            raise ValueError('History representation changed within a skill')
        p = float(step.completion_probability)
        if not math.isfinite(p) or not 0 <= p <= 1 or type(step.consistent) is not bool:
            raise ValueError('Invalid learned completion signal')
        self.streak = self.streak + 1 if p >= self.threshold and step.consistent else 0
        completed = self.streak >= self.confirmation_steps
        reward = self.ledger.observe(self.skill,confirmed=completed)
        self.reward += self.gamma ** self.duration * reward
        self.duration += 1
        self.next_history = history
        if step.terminated:
            self.ended = True
            return self._finish('terminated',True,completed)
        if completed:
            return self._finish('completed',False,True)
        if step.timeout or self.duration == self.max_steps:
            return self._finish('timeout',False,False)
        return None

    def _finish(self, reason, terminated, completed):
        self.active = False
        if self.duration == 0:
            return None  # first transition cut: no fabricated zero-duration sample
        return MacroTransition(self.skill,self.history,self.next_history,self.reward,
            self.gamma**self.duration,self.duration,reason,terminated,completed,self.achieved_before,tuple(sorted(self.ledger.achieved)))


@dataclass
class FixedSkillRollout:
    transitions: list
    actions: list
    model_calls: int
    achieved: tuple
    stop_reason: str


@torch.no_grad()
def collect_fixed_skills(backend, actor, root, schedule, thresholds, *, seed, generator,
                         max_model_calls, max_skill_steps, gamma=.997, confirmation_steps=2):
    """Explicit schedule of (policy task ID, skill name); never reset WM between skills.

    Caller must bind calibrated thresholds and real integration evidence before a
    training run. This collector neither reads inventory nor claims task success.
    """
    schedule = tuple(schedule)
    if not schedule or type(max_model_calls) is not int or max_model_calls < 1 or type(max_skill_steps) is not int or max_skill_steps < 1:
        raise ValueError('Nonempty schedule and positive integer budgets required')
    if root.episode.origin != 'offline' or root.episode.split != 'train':
        raise ValueError('Only offline train roots may be imagined')
    contract = getattr(backend,'contract',None)
    if contract is None:
        raise ValueError('Fixed skill execution requires a bound multi-task history contract')
    for task,name in schedule:
        contract.reward_task(task)
        if contract.policy_tasks[task] != name or task not in thresholds:
            raise ValueError('Schedule task names/thresholds do not match the contract')
        if not math.isfinite(thresholds[task]) or not 0 <= thresholds[task] <= 1:
            raise ValueError('Invalid completion threshold')
    executor = SkillExecutor(root.episode.episode_id,gamma=gamma,confirmation_steps=confirmation_steps)
    history = backend.reset(root,schedule[0][0],seed)
    transitions,actions,calls = [],[],0
    reason = 'schedule_complete'
    for index,(task,name) in enumerate(schedule):
        if calls >= max_model_calls:
            reason = 'budget_exhausted'; break
        if index:
            history = backend.begin_skill(task)
        executor.begin(name,history,threshold=thresholds[task],max_steps=min(max_skill_steps,max_model_calls-calls))
        result = None
        while executor.active:
            tid = torch.tensor(task,device=history.device,dtype=torch.long)
            scores = actor(history,tid)
            action = torch.stack([torch.multinomial(x.softmax(-1),1,generator=generator)[0] for x in scores])
            step = backend.step(action);calls += 1;actions.append(action.detach().cpu())
            result = executor.advance(step)
            if not step.uncertainty_cut:
                history = executor.next_history
        if result is not None:
            if result.reason == 'completed':
                result = MacroTransition(**{**vars(result),'next_history':backend.decision_history()})
            transitions.append(result)
        if result is None or result.reason != 'completed':
            reason = 'uncertainty_cut' if result is None else result.reason
            break
    return FixedSkillRollout(transitions,actions,calls,tuple(sorted(executor.ledger.achieved)),reason)


def feasible_logits(logits, feasibility, impossible, *, penalty, fallback):
    """Soft penalties for uncertain skills; caller supplies only high-confidence exclusions.

    fallback is an explicit observation/reselection action, not an inferred skill.
    If all skills are blocked, expose exactly this action and report the fallback.
    """
    if logits.shape != feasibility.shape or logits.shape != impossible.shape or logits.ndim < 1:
        raise ValueError('Feasibility shapes must match logits')
    if impossible.dtype != torch.bool or type(fallback) is not int or not 0 <= fallback < logits.shape[-1]:
        raise ValueError('Boolean exclusions and explicit fallback action required')
    if not math.isfinite(penalty) or penalty < 0 or not torch.isfinite(logits).all() or not torch.isfinite(feasibility).all() or not ((feasibility >= 0)&(feasibility <= 1)).all():
        raise ValueError('Invalid feasibility values')
    scores = logits - penalty*(1-feasibility.detach())
    blocked = impossible.all(-1)
    mask = impossible.clone()
    mask[...,fallback] &= ~blocked
    return scores.masked_fill(mask,-torch.inf),blocked
