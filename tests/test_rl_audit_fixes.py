import unittest
from types import SimpleNamespace
import torch
from dreamer_agent.imagination import ImaginationTrainer, RLConfig, ModelStep
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.contracts import Episode
from dreamer_agent.rl_data import OfflineBatch
from dreamer_agent.train_imagination import run, check_evidence


class Backend:
    frozen_modules=()
    def reset(self,*args): return torch.zeros(4)
    def step(self,action): return ModelStep(torch.zeros(4),.99,True)


class TestAuditFixes(unittest.TestCase):
    def trainer(self, seed=7):
        bc=HistoryPolicy(3,2,1,(2,),hidden=4).eval().requires_grad_(False)
        return ImaginationTrainer(bc,Backend(),RLConfig(bc_weight=0),{0:.9},seed=seed)

    def test_seed_controls_value(self):
        a,b=self.trainer(),self.trainer()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.value.parameters(),b.value.parameters())))
        c=self.trainer(8)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.value.parameters(),c.value.parameters())))

    def test_initialization_does_not_change_rng(self):
        bc=HistoryPolicy(3,2,1,(2,),hidden=4).eval().requires_grad_(False)
        state=torch.random.get_rng_state().clone()
        ImaginationTrainer(bc,Backend(),RLConfig(),{0:.9},seed=8)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_completed_objective_no_bootstrap(self):
        t=self.trainer()
        for p in t.value.parameters(): p.data.zero_()
        t.value.net[-1].bias.data.fill_(10)
        root=SimpleNamespace(episode=Episode('clip','session','train','offline'))
        tr,_=t.collect(root,0,seed=3)
        self.assertEqual(len(tr.rewards),2)
        self.assertFalse(tr.terminated.any())
        self.assertTrue(tr.rewarded[-1])
        self.assertEqual(tr.confirmation.tolist(),[0.,.5])
        result=t.update(tr,None)
        # Last target exactly 1 rather than 1+gamma*10. Penultimate lambda target.
        target0=.997*((1-.95)*10+.95*1)
        expected=((10-target0)**2+81)/2
        self.assertAlmostEqual(result['value_loss'],expected,places=4)

    def test_plain_bc_tuple_rejected(self):
        t=self.trainer();t.config=RLConfig()
        root=SimpleNamespace(episode=Episode('clip','session','train','offline'))
        tr,_=t.collect(root,0,seed=0)
        with self.assertRaisesRegex(ValueError,'OfflineBatch'):
            t.update(tr,(1,2,3,4))

    def test_eval_bc_rejected(self):
        b=OfflineBatch(torch.zeros(1,1,4),torch.zeros(1,1,dtype=torch.long),
          torch.zeros(1,1,1,1,dtype=torch.long),torch.ones(1,1,1,dtype=torch.bool),
          (Episode('a','b','test','real_evaluation'),),'x')
        with self.assertRaises(ValueError): b.validate('x')

    def test_public_run_cannot_skip_gate(self):
        with self.assertRaisesRegex(ValueError,'Verified evidence'):
            run(self.trainer(),[],[], '/unused',1,{})

    def test_unreviewed_config_rejected(self):
        import json,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json'
            p.write_text(json.dumps(dict(schema_version=1,stage='P2',approved_horizons=[16])))
            with self.assertRaisesRegex(ValueError,'Full RL configuration'):
                check_evidence(p,RLConfig())
