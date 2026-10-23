import unittest
from types import SimpleNamespace
import torch
from dreamer_agent.proxy_rl_pilot import ProxyPilotTrainer
from dreamer_agent.imagination import RLConfig,ModelStep
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.contracts import Episode
class Backend:
 frozen_modules=()
 def reset(self,*args):return torch.zeros(4)
 def step(self,a):return ModelStep(torch.zeros(4),.25,False)
class TestProxyPilot(unittest.TestCase):
 def make(self):
  bc=HistoryPolicy(3,2,1,(2,),hidden=4).eval().requires_grad_(False)
  trainer=ProxyPilotTrainer(bc,Backend(),RLConfig(bc_weight=0),{0:.5})
  root=SimpleNamespace(episode=Episode('clip','session','train','offline'))
  return trainer,root
 def test_terminal_proxy_without_fake_completion(self):
  t,r=self.make();tr,_=t.collect(r,0,seed=0)
  self.assertEqual(tr.rewards.tolist(),[0.]*15+[.25])
  self.assertFalse(tr.terminated.any())
  self.assertTrue(tr.boundary[-1])
  metrics=t.update(tr,None)
  self.assertEqual(metrics['reward_source'],'uncalibrated_terminal_proxy')
  self.assertNotIn('objective_complete',metrics)
  self.assertFalse(metrics['task_success_measured'])
 def test_nonfinite_proxy_fails(self):
  t,r=self.make()
  t.backend.step=lambda a:ModelStep(torch.zeros(4),float('nan'),False)
  with self.assertRaises(ValueError):t.collect(r,0,seed=0)
