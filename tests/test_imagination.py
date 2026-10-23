from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
import torch
from dreamer_agent.learning import HistoryPolicy, lambda_returns
from dreamer_agent.contracts import Episode
from dreamer_agent.rl_data import OfflineBatch
from dreamer_agent.imagination import RLConfig, ImaginationTrainer, ModelStep


class Backend:
    frozen_modules = ()
    def __init__(self, cut=None, end=None):
        self.cut, self.end = cut, end
    def reset(self, root, task, seed):
        self.i = 0
        return torch.zeros(4)
    def step(self, action):
        self.i += 1
        return ModelStep(torch.full((4,), self.i / 10), .99, True,
            uncertainty_cut=self.i == self.cut, timeout=self.i == self.end)


def setup(backend=None, config=None):
    torch.manual_seed(7)
    bc = HistoryPolicy(3, 2, 1, (2,), hidden=4).eval().requires_grad_(False)
    trainer = ImaginationTrainer(bc, backend or Backend(), config or RLConfig(), {0: .9}, data_binding="test-binding")
    root = SimpleNamespace(episode=Episode('clip', 'session', 'train', 'offline'))
    return trainer, root


def test_update_frozen_and_fresh():
    t, root = setup()
    snapshots = [[p.clone() for p in m.parameters()] for m in t.frozen]
    trajectory, _ = t.collect(root, 0, seed=1)
    assert trajectory.rewards.sum() == 1
    assert not trajectory.terminated.any()
    assert trajectory.boundary[-1]
    assert len(trajectory.rewards) == 2
    offline = OfflineBatch(torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, 8, 1, dtype=torch.long), torch.ones(1, 2, 8, dtype=torch.bool), (root.episode,), "test-binding")
    before = [p.clone() for p in t.actor.parameters()]
    assert t.update(trajectory, offline)['update'] == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, t.actor.parameters()))
    for m, original in zip(t.frozen, snapshots):
        assert all(torch.equal(a, b) and b.grad is None for a, b in zip(original, m.parameters()))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'fresh'):
        t.update(trajectory, offline)


def test_cut_discards_transition_and_bounds_prefix():
    t, root = setup(Backend(cut=3), RLConfig(confirmation_steps=4))
    trajectory, metrics = t.collect(root, 0, seed=1)
    assert len(trajectory.rewards) == 2
    assert trajectory.bounded_bootstrap.tolist() == [False, True]
    assert trajectory.boundary[-1] and not trajectory.terminated.any()
    assert metrics['model_calls'] == 3
    t, root = setup(Backend(cut=1))
    assert t.collect(root, 0, seed=1)[0] is None


def test_timeout_and_termination_returns():
    reward = torch.tensor([[1.], [100.]])
    nv = torch.tensor([[2.], [3.]], requires_grad=True)
    boundary = torch.ones(2, 1, dtype=torch.bool)
    terminal = torch.tensor([[False], [True]])
    ret = lambda_returns(reward, nv, torch.full_like(reward, .5), terminal, boundary)
    assert ret.tolist() == [[2.], [100.]]
    assert not ret.requires_grad


def test_eval_root_rejected_and_empty_pmpo_groups():
    t, root = setup(config=RLConfig(bc_weight=0))
    root.episode = Episode('clip', 'session', 'test', 'real_evaluation')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'offline training'):
        t.collect(root, 0, seed=1)
    from dreamer_agent.learning import pmpo_loss
    for sign in (-1., 1.):
        x = torch.zeros(3, 2, requires_grad=True)
        loss = pmpo_loss([x], [x.detach()], torch.zeros(3, 1, dtype=torch.long), torch.full((3,), sign))
        loss.backward()
        assert torch.isfinite(x.grad).all()


def test_missing_evidence_rejects():
    import json
    from dreamer_agent.train_imagination import check_evidence
    temporary = tempfile.TemporaryDirectory()
    p = Path(temporary.name)/'evidence.json'
    p.write_text(json.dumps(dict(schema_version=1, stage='P2', approved_horizons=[16])))
    with unittest.TestCase().assertRaises(ValueError):
        check_evidence(p, RLConfig())


class TestImagination(unittest.TestCase):
    pass

for name, function in list(globals().items()):
    if name.startswith("test_") and callable(function):
        setattr(TestImagination, name, lambda self, f=function: f())
