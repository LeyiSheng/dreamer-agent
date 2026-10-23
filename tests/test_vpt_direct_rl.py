import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from dreamer_agent.contracts import Episode
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.imagination import RLConfig,ModelStep
from dreamer_agent.vpt_direct_rl import DirectTrainer,save_checkpoint,restore_checkpoint,root_index

class Backend:
    frozen_modules=()
    def reset(self,root,task,seed):self.i=0;return torch.zeros(4)
    def step(self,action):
        self.i+=1;return ModelStep(torch.full((4,),self.i/20),.3,False)

def setup():
    torch.manual_seed(7)
    bc=HistoryPolicy(3,2,1,(2,),hidden=4).eval().requires_grad_(False)
    return DirectTrainer(bc,Backend(),RLConfig(horizon=16,bc_weight=0),{0:.5},seed=2)

class DirectRLTests(unittest.TestCase):
    def root(self):return SimpleNamespace(episode=Episode('clip','session','train','offline'))
    def test_no_supervised_updates_or_frozen_changes(self):
        t=setup();snap=[[p.clone() for p in m.parameters()] for m in t.frozen]
        tr,_=t.collect(self.root(),0,seed=0)
        self.assertEqual(tr.confirmation[0],0);self.assertEqual(tr.next_confirmation[-1],1)
        self.assertEqual(tr.rewards[:-1].sum(),0);self.assertFalse(tr.terminated.any())
        metrics=t.update(tr,None);self.assertEqual(metrics['bc_loss'],0)
        self.assertFalse(metrics['task_success_measured'])
        for m,old in zip(t.frozen,snap):
            for p,q in zip(m.parameters(),old):self.assertTrue(torch.equal(p,q));self.assertIsNone(p.grad)
    def test_exact_resume_next_actions_and_update(self):
        a=setup();tr,_=a.collect(self.root(),0,seed=0);a.update(tr,None)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'c.pt';save_checkpoint(path,a,{'test':1})
            b=setup();restore_checkpoint(path,b,{'test':1})
            ta,_=a.collect(self.root(),0,seed=1);tb,_=b.collect(self.root(),0,seed=1)
            self.assertTrue(torch.equal(ta.actions,tb.actions))
            a.update(ta,None);b.update(tb,None)
            for ma,mb in [(a.actor,b.actor),(a.value,b.value)]:
                for p,q in zip(ma.parameters(),mb.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            with self.assertRaises(ValueError):restore_checkpoint(path,b,{'test':2})
    def test_roots_cover_every_cycle_and_reject_eval(self):
        for c in range(3):self.assertEqual(sorted(root_index(i,2,8) for i in range(c*8,(c+1)*8)),list(range(8)))
        r=self.root();r.episode=Episode('test','session','test','real_evaluation')
        with self.assertRaises(ValueError):setup().collect(r,0,seed=1)
if __name__=='__main__':unittest.main()
