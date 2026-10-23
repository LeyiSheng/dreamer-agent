import unittest
import torch
from torch import nn
from dreamer_agent.wm_visual import cache_with_current
from dreamer_agent.wm_finetune import (
    training_objective, validate_resume_objective, clip_loss_v2,
    load_lora_state, LoRALinear,
)


class VisualContractTests(unittest.TestCase):
    def test_entry_is_versioned_and_ambiguous_resume_rejected(self):
        self.assertIs(training_objective('shortcut_v2'), clip_loss_v2)
        validate_resume_objective({'objective': 'v2'}, 'shortcut_v2')
        for config in ({}, {'objective': 'legacy_denoise'}):
            with self.assertRaises(ValueError):
                validate_resume_objective(config, 'shortcut_v2')

    def test_cache_current_gradient_past_detached_and_mutation_safe(self):
        history = torch.arange(12.).reshape(4, 3).requires_grad_()
        current = torch.ones(1, 3, requires_grad=True)
        result = cache_with_current(history, current, 2)
        with torch.no_grad():
            history.zero_()
        result.square().sum().backward()
        self.assertIsNone(history.grad)
        torch.testing.assert_close(current.grad, torch.full_like(current, 2.))
        torch.testing.assert_close(result[1], torch.tensor([3., 4., 5.]))

    def test_ring_cache_preserves_physical_slots(self):
        history = torch.arange(12.).reshape(4, 3)
        current = torch.full((1, 3), 42.)
        result = cache_with_current(history, current, 5)
        expected = history.clone();expected[1] = 42.
        torch.testing.assert_close(result, expected)

    def test_lora_load_must_be_complete_finite_and_shape_compatible(self):
        layer = LoRALinear(nn.Linear(3, 2), 1, 2)
        state = {k: p.detach().clone() for k, p in layer.named_parameters() if p.requires_grad}
        load_lora_state(layer, state)
        for broken in ({'A': state['A']}, {**state, 'B': torch.zeros(3, 1)},
                       {**state, 'B': torch.full_like(state['B'], float('nan'))}):
            with self.assertRaises(ValueError):load_lora_state(layer, broken)


if __name__ == '__main__':
    unittest.main()
