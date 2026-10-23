import unittest
import torch
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.reward import RewardCompletionHeads, supervised_head_loss, frozen_history


class RewardTest(unittest.TestCase):
    def test_reward_gradient_cannot_reach_history(self):
        head = RewardCompletionHeads(4, 2, 8)
        h = torch.randn(2, 3, 4, requires_grad=True)
        nxt = torch.randn_like(h, requires_grad=True)
        scores = head(h, nxt, torch.zeros(2, 3, dtype=torch.long))
        labels = {k: torch.ones(2, 3) for k in scores}
        masks = {k: torch.ones(2, 3, dtype=torch.bool) for k in scores}
        supervised_head_loss(scores, labels, masks).backward()
        self.assertIsNone(h.grad)
        self.assertIsNone(nxt.grad)
        self.assertGreater(head.transition.weight.grad.abs().sum().item(), 0)

    def test_unknown_labels_are_not_negatives(self):
        scores = {k: torch.zeros(2, requires_grad=True) for k in ("event", "completion")}
        labels = {k: torch.tensor([1., float("nan")]) for k in scores}
        masks = {k: torch.tensor([True, False]) for k in scores}
        loss = supervised_head_loss(scores, labels, masks)
        self.assertAlmostEqual(loss.item(), 0.69314718, places=6)
        loss.backward()
        for score in scores.values():
            self.assertEqual(score.grad[1].item(), 0)
            self.assertLess(score.grad[0].item(), 0)

    def test_event_and_completion_have_independent_targets(self):
        scores = {k: torch.zeros(1, requires_grad=True) for k in ("event", "completion")}
        labels = dict(event=torch.ones(1), completion=torch.zeros(1))
        masks = {k: torch.ones(1, dtype=torch.bool) for k in scores}
        supervised_head_loss(scores, labels, masks).backward()
        self.assertLess(scores["event"].grad.item(), 0)
        self.assertGreater(scores["completion"].grad.item(), 0)

    def test_empty_or_invalid_supervision_rejected(self):
        scores = {k: torch.zeros(1) for k in ("event", "completion")}
        labels = {k: torch.ones(1) * 2 for k in scores}
        for enabled in (False, True):
            with self.assertRaises(ValueError):
                supervised_head_loss(scores, labels, {k: torch.tensor([enabled]) for k in scores})

    def test_frozen_history_is_causal_and_requires_freeze(self):
        policy = HistoryPolicy(3, 2, 2, (2,), hidden=4, targets=2)
        x, a = torch.randn(1, 5, 3), torch.randn(1, 5, 2)
        with self.assertRaises(ValueError):
            frozen_history(policy, x, a)
        policy.eval().requires_grad_(False)
        h = frozen_history(policy, x, a)
        changed = x.clone()
        changed[:, 3:] += 50
        other = frozen_history(policy, changed, a)
        self.assertTrue(torch.equal(h[:, :3], other[:, :3]))
        self.assertFalse(h.requires_grad)

    def test_supervised_fixture_can_learn_both_heads(self):
        torch.manual_seed(47)
        head = RewardCompletionHeads(2, 1, 8)
        h = torch.tensor([[-1.,0.], [1.,0.], [-2.,0.], [2.,0.]])
        y = dict(event=torch.tensor([0.,1.,0.,1.]), completion=torch.tensor([1.,0.,1.,0.]))
        masks = {k: torch.ones(4, dtype=torch.bool) for k in y}
        task = torch.zeros(4, dtype=torch.long)
        opt = torch.optim.Adam(head.parameters(), lr=.03)
        initial = supervised_head_loss(head(h,h,task),y,masks).item()
        for _ in range(60):
            opt.zero_grad()
            loss = supervised_head_loss(head(h,h,task),y,masks)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), initial * .1)


if __name__ == "__main__":
    unittest.main()
