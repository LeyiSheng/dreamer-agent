import unittest
import torch
from dreamer_agent.reward import TemporalRewardCompletionHeads
class TemporalRewardTests(unittest.TestCase):
 def test_causal_and_reset(self):
  torch.manual_seed(7);m=TemporalRewardCompletionHeads(4,2,8).eval()
  x=torch.randn(1,40,4);n=torch.randn_like(x);t=torch.zeros(1,40,dtype=torch.long)
  a=m(x,n,t);changed=x.clone();changed[:,20:]+=100
  b=m(changed,n,t)
  for k in a: self.assertTrue(torch.equal(a[k][:,:20],b[k][:,:20]))
  m(changed,n,t)
  for k,v in m(x,n,t).items():self.assertTrue(torch.equal(v,a[k]))
 def test_memory_survives_bc_boundary(self):
  torch.manual_seed(7);m=TemporalRewardCompletionHeads(4,2,8).eval()
  x=torch.randn(1,40,4);n=torch.randn_like(x);t=torch.zeros(1,40,dtype=torch.long)
  a=m(x,n,t);short=m(x[:,32:],n[:,32:],t[:,32:])
  self.assertFalse(torch.equal(a["completion"][:,32],short["completion"][:,0]))
 def test_frozen_inputs_and_gradients(self):
  m=TemporalRewardCompletionHeads(4,2,8);x=torch.randn(1,40,4,requires_grad=True)
  r=m(x,x,torch.zeros(1,40,dtype=torch.long));sum(v.sum() for v in r.values()).backward()
  self.assertIsNone(x.grad);self.assertTrue(all(p.grad is not None for p in m.parameters()))
if __name__=="__main__":unittest.main()
