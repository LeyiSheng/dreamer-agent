"""Real-weight decoder parity and LoRA gradient gate; zero optimizer updates."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from . import wm_finetune as wm
from .wm_visual import enable_decoder_grad, decode_frame, sample_last_grad, visual_losses, load_perceptual
from .provenance import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.60)
    torch.manual_seed(928)
    tokenizer, dynamics, _, _ = wm.load_backends('cuda:0')
    with np.load(args.clip) as data:
        frames = data['frames'].copy()
        actions = torch.tensor(data['actions'].astype(np.int64), device='cuda:0')
    if len(frames) <= tokenizer.decoder.window_size:
        raise ValueError('Gate clip must cross the decoder cache window')
    z = wm._encode(tokenizer, frames, 'cuda:0')
    shape = (1, 1, 3, 384 // tokenizer.patch_size[0], 640 // tokenizer.patch_size[1])
    tokenizer.decoder.init_rope_cache(shape)
    tokenizer.decoder.clean_kvcache()
    native = []
    with torch.no_grad():
        for i in range(len(frames)):native.append(decode_frame(tokenizer, z[:, i:i+1], i).cpu())
    enable_decoder_grad(tokenizer)
    tokenizer.decoder.clean_kvcache()
    maximum = 0.;mean = 0.
    for i in range(len(frames)):
        latent = z[:, i:i+1].detach().clone().requires_grad_()
        reconstructed = decode_frame(tokenizer, latent, i)
        difference = (reconstructed.detach().cpu() - native[i]).abs()
        maximum = max(maximum, float(difference.max()));mean = max(mean, float(difference.mean()))
        reconstructed.float().square().mean().backward()
        assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().max() > 0
    report = dict(clip=str(args.clip),clip_sha256=sha256(args.clip),optimizer_steps=0,
        decoder_max_abs=maximum,decoder_max_frame_mae=mean,
        tolerances=dict(max_abs=.08,mean_abs=.005),decoder_gradient_passed=True,
        parity_frames=len(frames),decoder_window=tokenizer.decoder.window_size,
        code_hashes={n:sha256(Path(__file__).parent/n) for n in
                     ['wm_finetune.py','wm_visual.py','verify_wm_visual.py']})
    print('DECODER',json.dumps(report),flush=True)
    assert maximum <= .08 and mean <= .005, 'Decoder numerical parity failed'
    wm.install_lora(dynamics, last_n=4, rank=16, alpha=32)
    perceptual = load_perceptual('cuda:0')
    dynamics.clean_kvcache();tokenizer.decoder.clean_kvcache()
    generator = torch.Generator(device='cuda:0').manual_seed(928)
    with torch.no_grad():
        for i in range(16):
            wm._denoise(dynamics,z[:,i:i+1],actions[i:i+1],i,wm.CONTEXT_T,generator,wm.CONTEXT_STRIDE)
            decode_frame(tokenizer,z[:,i:i+1],i)
    nxt = sample_last_grad(dynamics,z[:,16:17],actions[15:16],16,generator)
    pixels = decode_frame(tokenizer,nxt,16)
    real = torch.from_numpy(frames[16]).permute(2,0,1)[None].to('cuda:0').float()/127.5-1
    l1,lp = visual_losses(pixels,real,perceptual)
    (l1+lp).backward()
    grads = {n:float(v.grad.float().norm()) for n,v in dynamics.named_parameters() if v.requires_grad and v.grad is not None}
    assert grads and any(v>0 for v in grads.values()) and all(np.isfinite(v) for v in grads.values())
    assert all(p.grad is None for p in tokenizer.parameters())
    assert all(p.grad is None for p in perceptual.parameters())
    assert all(p.grad is None for p in dynamics.parameters() if not p.requires_grad)
    report.update(passed=True,l1=float(l1.detach()),lpips=float(lp.detach()),
        lora_grad_nonzero=sum(v>0 for v in grads.values()),lora_grad_tensors=len(grads),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        frozen_gradient_leak=False,gradient_scope='final denoising call, current-frame decoder; detached past')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print('GATE',json.dumps(report),flush=True)


if __name__ == '__main__':main()
