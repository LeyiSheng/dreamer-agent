import unittest
import torch
from dreamer_agent.contracts import Episode, MilestoneLedger, validate_dataset
from dreamer_agent.learning import HistoryPolicy, bc_loss, lambda_returns, macro_reward, pmpo_loss


class ContractsTest(unittest.TestCase):
    def test_session_leakage(self):
        with self.assertRaises(ValueError):
            validate_dataset([Episode('clip1', 'session', 'train', 'offline'),
                              Episode('clip2', 'session', 'test', 'offline')])

    def test_evaluation_and_synthetic_rejected(self):
        for origin in ['real_evaluation', 'synthetic']:
            with self.assertRaises(ValueError):
                validate_dataset([Episode('one', 'session', 'train', origin)])

    def test_valid_clips(self):
        validate_dataset([Episode('a', 'one', 'train', 'offline'),
                          Episode('b', 'one', 'train', 'offline')])

    def test_rewards_are_episode_scoped(self):
        ledger = MilestoneLedger()
        self.assertEqual(ledger.observe('wood', confirmed=False), 0)
        self.assertEqual(ledger.observe('wood', confirmed=True), 1)
        self.assertEqual(ledger.observe('table', confirmed=True), 1)
        self.assertEqual(ledger.observe('wood', confirmed=True), 0)


class LearningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_timeout_bootstraps_without_cross_episode_leak(self):
        r = torch.tensor([[1.], [100.]])
        v = torch.tensor([[10.], [20.]], requires_grad=True)
        d = torch.full_like(r, .5)
        out = lambda_returns(r, v, d, torch.tensor([[False], [True]]),
                             torch.tensor([[True], [True]]), lam=1)
        torch.testing.assert_close(out, torch.tensor([[6.], [100.]]))
        self.assertFalse(out.requires_grad)

    def test_skill_boundary_is_not_terminal(self):
        r = torch.tensor([[1.], [2.]])
        v = torch.zeros_like(r)
        out = lambda_returns(r, v, torch.full_like(r, .5),
                             torch.tensor([[False], [True]]),
                             torch.tensor([[False], [True]]), lam=1)
        torch.testing.assert_close(out, torch.tensor([[2.], [2.]]))

    def test_duration_discount(self):
        r, d = macro_reward(torch.tensor([1., 2., 4.]), .5)
        self.assertEqual(r.item(), 3.)
        self.assertEqual(d, .125)

    def test_bc_mask_and_gradient(self):
        model = HistoryPolicy(4, 2, 3, (2, 3), hidden=8, targets=2)
        logits = model(torch.randn(2, 3, 4), torch.zeros(2, 3, 2), torch.zeros(2, 3, dtype=torch.long))
        targets = torch.zeros(2, 3, 2, 2, dtype=torch.long)
        mask = torch.ones(2, 3, 2, dtype=torch.bool)
        mask[:, -1, 1] = False
        targets[:, -1, 1] = -999  # invalid future targets must never be indexed
        loss = bc_loss(logits, targets, mask)
        loss.backward()
        self.assertGreater(model.history.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_history_is_causal(self):
        model = HistoryPolicy(4, 2, 3, (2,), hidden=8)
        x = torch.randn(1, 4, 4)
        a, tasks = torch.zeros(1, 4, 2), torch.zeros(1, 4, dtype=torch.long)
        before = model(x, a, tasks)[0]
        x[:, 2:] += 100
        torch.testing.assert_close(before[:, :2], model(x, a, tasks)[0][:, :2])

    def test_pmpo_sign_and_detached_prior(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        prior = torch.zeros(2, 2, requires_grad=True)
        advantage = torch.tensor([1., -1.], requires_grad=True)
        loss = pmpo_loss([logits], [prior], torch.zeros(2, 1, dtype=torch.long), advantage)
        loss.backward()
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[1, 0].item(), 0)
        self.assertIsNone(prior.grad)
        self.assertIsNone(advantage.grad)

    def test_empty_sign_groups_are_finite(self):
        for sign in [1., -1.]:
            loss = pmpo_loss([torch.zeros(3, 2)], [torch.zeros(3, 2)],
                             torch.zeros(3, 1, dtype=torch.long), torch.full((3,), sign))
            self.assertTrue(torch.isfinite(loss))

    def test_exact_kl(self):
        p, q = torch.tensor([[.8, .2]]), torch.tensor([[.4, .6]])
        actions, adv = torch.zeros(1, 1, dtype=torch.long), torch.ones(1)
        a = pmpo_loss([p.log()], [q.log()], actions, adv, beta=1)
        b = pmpo_loss([p.log()], [q.log()], actions, adv, beta=0)
        torch.testing.assert_close(a-b, (p*(p/q).log()).sum())


if __name__ == '__main__':
    unittest.main()
