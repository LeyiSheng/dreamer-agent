import unittest
import torch
from dreamer_agent.rl_pixels import generated_to_p1_pixels
class TestPixels(unittest.TestCase):
 def test_padding_and_quantization(self):
  frame=torch.full((1,1,3,384,640),.25)
  out=generated_to_p1_pixels(frame)
  self.assertEqual(torch.count_nonzero(out[:,:,:,:12]),0)
  self.assertEqual(torch.count_nonzero(out[:,:,:,-12:]),0)
  expected=torch.tensor(int(1.25*127.5)/127.5-1)
  self.assertTrue(torch.allclose(out[:,:,:,12:-12],expected))
 def test_nan_rejected(self):
  with self.assertRaises(ValueError):generated_to_p1_pixels(torch.full((1,1,3,384,640),float('nan')))
