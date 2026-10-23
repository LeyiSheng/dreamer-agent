"""Canonical P1 preprocessing for WM decoded pixels, with zero padding restored."""
import torch


def generated_to_p1_pixels(frame):
    if frame.shape != (1,1,3,384,640) or not torch.isfinite(frame).all():
        raise ValueError('Expected one finite normalized decoded frame')
    # Recorded P1 inputs are uint8 RGB with zero normalized vertical padding.
    rgb=((frame[:,:, :,12:-12,:].float().clamp(-1,1)+1)*127.5).to(torch.uint8)
    return torch.nn.functional.pad(rgb.float()/127.5-1,(0,0,12,12))


class P1PixelEncoder:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer
    @torch.no_grad()
    def __call__(self,pixels):
        device=next(self.tokenizer.parameters()).device
        if device.type=='cuda' and torch.cuda.current_device()!=device.index:
            raise ValueError('Select the tokenizer CUDA device before inference')
        if pixels.shape!=(1,1,3,384,640) or not torch.isfinite(pixels).all():
            raise ValueError('Invalid P1 pixels')
        if torch.count_nonzero(pixels[:,:,:,:12]) or torch.count_nonzero(pixels[:,:,:,-12:]):
            raise ValueError('P1 padding must be normalized zero')
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            feature=self.tokenizer.encoder(pixels.to(device))
        feature=feature.float().reshape(-1)
        if not torch.isfinite(feature).all():raise ValueError('Nonfinite P1 feature')
        return feature
