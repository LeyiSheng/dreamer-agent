import unittest
from types import SimpleNamespace
import torch
from torch import nn
from dreamer_agent.actions import ActionSpec
from dreamer_agent.world_model import FrozenWorldAdapter


class Cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k", torch.zeros(16, 1))
        self.register_buffer("v", torch.zeros(16, 1))
        self.cached_frames = 0
    def clean_cache(self):
        self.k.zero_()
        self.v.zero_()


class Encoder(nn.Module):
    num_learnable_tokens = 4
    def forward(self, pixels):
        values = pixels.mean((2, 3, 4))
        return values[..., None, None].expand(1, pixels.shape[1], 4, 1).clone()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = Cache()
        self.calls = []
    def init_rope_cache(self, shape):
        self.shape = shape


class Tokenizer(nn.Module):
    patch_size = (1, 1)
    config = SimpleNamespace(window_size=16)
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()
    def ar_decode(self, tokens, shape, current_frame_idx):
        assert tokens.shape[:2] == (1, 1), "wrong time axis"
        index = int(current_frame_idx)
        self.decoder.calls.append((index, tokens.clone()))
        self.decoder.cache.k[index % 16] = tokens.mean()
        return tokens.mean().expand(1, 1, 3, 2, 2)


class Dynamics(nn.Module):
    config = SimpleNamespace(K_samples_step=64, window_size=16, in_dim=2)
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.cache = Cache()
        self.calls = []
    def init_rope_cache(self, **kwargs):
        pass
    def get_read_rope_indices(self, i):
        return torch.arange(16)
    def forward(self, img, action_ids, current_frame_idx, **kwargs):
        assert img.shape[:2] == (1, 1)
        assert action_ids.shape == (1, 2)
        index = int(current_frame_idx)
        self.calls.append((index, action_ids.clone()))
        self.cache.k[index % 16] = img.mean() + action_ids.sum()
        return img


def sampler(model, x, frame_idx, action_ids, steps_size, **kwargs):
    model.last_steps = steps_size
    previous = model.cache.k.sum().clone()
    model.cache.k[frame_idx % 16] = x.mean()
    return x + previous + action_ids.sum()


class AdapterTest(unittest.TestCase):
    def make(self, steps=4):
        spec = ActionSpec(((7,), (2, 3)), "a"*64, 20., 1)
        return FrozenWorldAdapter(Tokenizer(), Dynamics(), sampler, spec,
                                  image_size=(2, 2), steps=steps)

    def inputs(self, count=3):
        p = torch.arange(count, dtype=torch.float32).reshape(1, count, 1, 1, 1).expand(1, count, 3, 2, 2)/count
        a = torch.tensor([[7, 2+i % 2] for i in range(count)])
        return p, a

    def test_each_frame_gets_its_own_action_and_latent(self):
        for count in (1, 2, 4):
            model = self.make()
            pixels, actions = self.inputs(count)
            model.reset(pixels, actions)
            self.assertEqual(model.frame_index, count)
            for i, (index, action) in enumerate(model.dynamics.calls):
                self.assertEqual(index, i)
                torch.testing.assert_close(action, actions[i:i+1])
                self.assertAlmostEqual(model.tokenizer.decoder.calls[i][1].mean().item(), i/count)

    def test_reject_length_and_batch_before_cache_mutation(self):
        model = self.make()
        p, a = self.inputs()
        model.reset(p, a)
        before = model.dynamics.cache.k.clone()
        for pixels, actions in [(p, a[:2]), (p.expand(2, -1, -1, -1, -1), a)]:
            with self.assertRaises(ValueError):
                model.reset(pixels, actions)
            torch.testing.assert_close(model.dynamics.cache.k, before)

    def test_seed_reset_and_global_rng_isolation(self):
        model = self.make()
        p, a = self.inputs()
        global_state = torch.get_rng_state().clone()
        model.reset(p, a, seed=3)
        first = model.step(a[:1])[0]
        model.reset(p, a, seed=3)
        second = model.step(a[:1])[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))

    def test_branch_snapshot_restores_rng_and_caches(self):
        model = self.make()
        p, a = self.inputs()
        model.reset(p, a, seed=4)
        root = model.clone_state()
        original = {name:k.clone() for name,(k,v,c) in root.caches.items()}
        first = model.step(a[:1])[0]
        model.restore_state(root)
        model.step(a[1:2])
        model.restore_state(root)
        replay = model.step(a[:1])[0]
        torch.testing.assert_close(first, replay, rtol=0, atol=0)
        for name, values in root.caches.items():
            torch.testing.assert_close(values[0], original[name], rtol=0, atol=0)

    def test_steps_configuration_reaches_sampler(self):
        model = self.make(steps=2)
        p,a = self.inputs()
        model.reset(p,a)
        model.step(a[:1])
        self.assertEqual(model.dynamics.last_steps,2)

    def test_shared_mutable_backend_rejected(self):
        model = self.make()
        with self.assertRaisesRegex(ValueError, "already owned"):
            FrozenWorldAdapter(model.tokenizer, model.dynamics, sampler, model.action_spec, image_size=(2,2))

    def test_foreign_snapshot_rejected(self):
        one,two=self.make(),self.make()
        p,a=self.inputs()
        one.reset(p,a)
        with self.assertRaises(ValueError):
            two.restore_state(one.clone_state())

    def test_failure_requires_reset(self):
        model=self.make()
        p,a=self.inputs()
        model.reset(p,a)
        def fail(**kwargs):
            raise RuntimeError("backend failure")
        model.sampler=fail
        with self.assertRaises(RuntimeError):
            model.step(a[:1])
        with self.assertRaises(RuntimeError):
            model.clone_state()

if __name__ == "__main__":
    unittest.main()
