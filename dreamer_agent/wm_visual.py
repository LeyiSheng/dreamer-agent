"""Differentiable one-frame visual readout with detached past, for branch W.

This is perceptual supervised adaptation, not discrete-token GRPO. The sampler
backpropagates through its final denoising call only; earlier denoising steps,
previous frames and temporal Dynamics paths remain detached. Decoder current
Q/K/V are differentiable; frozen historical cache values are constants.
"""
from types import MethodType
import torch
import torch.nn.functional as F


def cache_with_current(cache, current, index):
    """Functional cache prefix: preserve gradients only for the current frame."""
    if current.shape != cache[:1].shape or index < 0:
        raise ValueError('Expected one frame with matching cache geometry')
    length = min(index + 1, len(cache))
    slot = index % len(cache)
    return torch.cat((cache[:slot].detach().clone(), current,
                      cache[slot + 1:length].detach().clone()), 0)


def decoder_temporal(self, x, freqs, shape, current_frame_idx=None, read_rope_indices=None):
    from src.modules.tokenizer import fast_rope_apply
    from flash_attn import flash_attn_func
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if b != 1 or shape[:2] != (1, 1) or current_frame_idx is None:
        raise ValueError('Differentiable decoder requires a single causal frame')
    index = int(current_frame_idx.item())
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    q, k = q.to(v.dtype), k.to(v.dtype)
    keys = cache_with_current(self.kvcache.k, k, index)
    values = cache_with_current(self.kvcache.v, v, index)
    length = len(keys)
    q = fast_rope_apply(q, self.kvcache.get_last_rope(current_frame_idx).view(-1, d // 2))
    kfreq = self.kvcache.get_rope(read_rope_indices)[:length].reshape(-1, d // 2)
    keys = fast_rope_apply(keys.reshape(1, length * s, n, d), kfreq)
    q = q.reshape(s, 1, n, d).contiguous()
    keys = keys.reshape(length, s, n, d).permute(1, 0, 2, 3).contiguous()
    values = values.permute(1, 0, 2, 3).contiguous()
    output = flash_attn_func(q, keys, values, dropout_p=0., causal=False)
    with torch.no_grad():
        self.kvcache.store_kvcache(k.detach(), v.detach(), current_frame_idx)
    return self.o(output.reshape(1, s, n * d))


def enable_decoder_grad(tokenizer):
    tokenizer.eval().requires_grad_(False)
    tokenizer.use_decoder_graph = False
    changed = []
    for block in tokenizer.decoder.blocks:
        module = block.attn_temporal
        if module is not None:
            changed.append((module, module.forward))
            module.forward = MethodType(decoder_temporal, module)
    if not changed:
        raise ValueError('No decoder temporal modules found')
    return changed


def restore_decoder(changed):
    for module, forward in changed:
        module.forward = forward


def sample_last_grad(dynamics, like, action, frame_idx, generator):
    from .wm_finetune import _pred, STRIDE_INDEX, K_SAMPLES
    noise = torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=generator)
    x = noise
    for i in range(4):
        with torch.set_grad_enabled(i == 3):
            predicted = _pred(dynamics, x.detach(), action, frame_idx, i * 16, STRIDE_INDEX)
            fraction = (i + 1) / 4
            x = predicted * fraction + noise * (1 - fraction)
    return x


def decode_frame(tokenizer, latent, index):
    shape = (1, 1, 3, 384 // tokenizer.patch_size[0], 640 // tokenizer.patch_size[1])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pixels = tokenizer.ar_decode(latent, shape=shape,
            current_frame_idx=torch.tensor([index], device=latent.device))
    return pixels[:, 0, :, 12:372].float()


def visual_losses(pred, real, perceptual):
    if pred.shape != real.shape or pred.ndim != 4 or pred.shape[-2:] != (360, 640):
        raise ValueError('Expected matched unpadded RGB frames in [-1,1]')
    l1 = (pred - real).abs().mean()
    # Keep whole-image L1; LPIPS at 180x320 is a registered cost compromise.
    small_pred = F.interpolate(pred, (180, 320), mode='bilinear', align_corners=False)
    small_real = F.interpolate(real, (180, 320), mode='bilinear', align_corners=False)
    lp = perceptual(small_pred, small_real).mean()
    return l1, lp


def load_perceptual(device):
    from pathlib import Path
    import lpips
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError('Pretrained AlexNet must already be available: ' + str(checkpoint))
    return lpips.LPIPS(net='alex', verbose=False).to(device).eval().requires_grad_(False)
