import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from dreamer_agent.learning import HistoryPolicy, bc_loss
from dreamer_agent.reward import TemporalRewardCompletionHeads
from dreamer_agent.imagination import ModelStep, RLConfig
from dreamer_agent.imagination_backend import FrozenFeatureBackend, OfflineRoot
from dreamer_agent.rl_contract import HistoryRewardContract, StreamingReadout, module_sha256
from dreamer_agent.rl_roots import tensor_hash
from dreamer_agent.provenance import sha256
from dreamer_agent.contracts import Episode
from dreamer_agent.skill_execution import SkillExecutor, collect_fixed_skills, feasible_logits


def models(context=32):
    torch.manual_seed(73)
    policy = HistoryPolicy(3,2,3,(2,),hidden=8,targets=8).eval().requires_grad_(False)
    reward = TemporalRewardCompletionHeads(8,2,8).eval().requires_grad_(False)
    contract = HistoryRewardContract(context,('wood','stone','general_behavior'),('stone','wood'),
        module_sha256(policy),module_sha256(reward),'a'*64,'b'*64)
    pm = dict(tasks=list(contract.policy_tasks),context=context)
    rm = dict(policy_state_sha256=contract.policy_state_sha256,tasks=list(contract.reward_tasks),context=context,
        feature_contract_sha256=contract.feature_contract_sha256,action_spec_sha256=contract.action_spec_sha256,
        history_protocol=contract.history_protocol,reward_protocol=contract.reward_protocol)
    return policy,reward,contract,pm,rm


class StreamingTests(unittest.TestCase):
    def test_contract_names_not_integer_coincidence(self):
        p,r,c,pm,rm=models(128)
        c.verify_training_metadata(pm,rm);c.verify_modules(p,r)
        self.assertEqual(c.reward_task(0),1)
        self.assertEqual(c.reward_task(1),0)
        with self.assertRaises(ValueError): c.reward_task(2)
        for bad in [dict(rm,context=32),dict(rm,tasks=['wood','stone']),dict(rm,policy_state_sha256='c'*64)]:
            with self.assertRaises(ValueError): c.verify_training_metadata(pm,bad)
        with self.assertRaises(ValueError): c.verify_training_metadata(dict(pm,tasks=['stone','wood','general_behavior']),rm)
        with torch.no_grad(): r.completion.bias.add_(.1)
        with self.assertRaises(ValueError): c.verify_modules(p,r)

    def test_contract_serialization_root_binding_and_frozen_state(self):
        p,r,c,pm,rm=models()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'contract.json';c.save(path)
            self.assertEqual(c,HistoryRewardContract.load(path))
        c.verify_root(dict(runtime_contract_sha256=c.fingerprint,feature_contract_sha256=c.feature_contract_sha256))
        with self.assertRaises(ValueError): c.verify_root(dict(runtime_contract_sha256='old'))
        p.train()
        with self.assertRaises(ValueError): StreamingReadout(p,r,c)

    def test_streaming_equals_full_prefix_across_both_context_boundaries(self):
        for context in (32,128):
            p,r,c,_,_=models(context)
            stream=StreamingReadout(p,r,c)
            features=torch.randn(context+5,3);past=torch.randn(context+5,2);past[0]=0
            h=stream.reset(features[0],past[0],0)
            torch.testing.assert_close(h,p.encode_history(features[:1][None],past[:1][None])[0,-1])
            pairs=[];begin=0
            for i in range(1,len(features)):
                old=h
                h,prob,boundary=stream.step(features[i],past[i])
                if i%context==0:begin=i
                expected=p.encode_history(features[begin:i+1][None],past[begin:i+1][None])[0,-1]
                torch.testing.assert_close(h,expected,rtol=1e-5,atol=1e-6)
                pairs.append((old,expected))
                before=torch.stack([x[0] for x in pairs])[None];after=torch.stack([x[1] for x in pairs])[None]
                score=r(before,after,torch.ones(1,len(pairs),dtype=torch.long))['completion'][0,-1].sigmoid()
                self.assertAlmostEqual(prob,float(score),places=6)
                self.assertEqual(boundary,i%context==0)
            reset=stream.reset(features[0],past[0],0)
            fresh=StreamingReadout(p,r,c);torch.testing.assert_close(reset,fresh.reset(features[0],past[0],0))
            self.assertIsNone(stream.reward_state)

    def test_streaming_recurrent_work_is_linear_and_instances_independent(self):
        p,r,c,_,_=models()
        count={'policy':0,'reward':0}
        def count_p(m,args):count['policy']+=args[0].shape[1]
        def count_r(m,args):count['reward']+=args[0].shape[1]
        hp=p.history.register_forward_pre_hook(count_p);hr=r.memory.register_forward_pre_hook(count_r)
        a=StreamingReadout(p,r,c);a.reset(torch.zeros(3),torch.zeros(2),0)
        for i in range(64):a.step(torch.full((3,),i/100),torch.zeros(2))
        hp.remove();hr.remove()
        self.assertEqual(count,dict(policy=65,reward=64))
        b=StreamingReadout(p,r,c);b.reset(torch.zeros(3),torch.zeros(2),0)
        prior=a.history.clone();b.step(torch.ones(3),torch.ones(2))
        torch.testing.assert_close(a.history,prior)


class ObjectiveTests(unittest.TestCase):
    def test_bc_normalizations_with_unequal_masks(self):
        x=torch.zeros(1,2,8,2,requires_grad=True)
        mask=torch.zeros(1,2,8,dtype=torch.bool);mask[0,0,:]=True;mask[0,1,0]=True
        target=torch.full((1,2,8,1),-999,dtype=torch.long);target[mask]=0
        mean=bc_loss([x],target,mask)
        summed=bc_loss([x],target,mask,normalization='anchor_sum')
        self.assertAlmostEqual(float(mean.detach()),math.log(2),places=6)
        self.assertAlmostEqual(float(summed.detach()),9*math.log(2)/2,places=6)
        summed.backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[~mask].abs().sum()),0.)
        with self.assertRaises(ValueError):RLConfig(bc_normalization='unknown')

    def test_soft_mask_and_explicit_all_blocked_fallback(self):
        scores,used=feasible_logits(torch.zeros(2,3),torch.tensor([[1.,.5,0.],[0.,0.,0.]]),
            torch.tensor([[False,False,True],[True,True,True]]),penalty=2.,fallback=1)
        torch.testing.assert_close(scores[0,:2],torch.tensor([0.,-1.]))
        self.assertEqual(used.tolist(),[False,True])
        torch.testing.assert_close(scores[1].softmax(-1),torch.tensor([0.,1.,0.]))


class SkillTests(unittest.TestCase):
    def test_duration_discount_completion_bootstrap_and_global_dedup(self):
        e=SkillExecutor('episode',gamma=.5)
        for expected in (.5,0.):
            e.begin('wood',torch.zeros(4),threshold=.9,max_steps=10)
            self.assertIsNone(e.advance(ModelStep(torch.ones(4),.99,True)))
            tr=e.advance(ModelStep(torch.ones(4)*2,.99,True))
            self.assertEqual(tr.reward,expected);self.assertEqual(tr.discount,.25)
            self.assertFalse(tr.terminated);self.assertTrue(tr.completed)
            self.assertEqual(float(tr.target(torch.tensor(4.,requires_grad=True))),expected+1)
            self.assertFalse(tr.target(torch.tensor(4.,requires_grad=True)).requires_grad)
            self.assertEqual(tr.achieved_after,('wood',))

    def test_timeout_terminal_and_uncertainty_prefix(self):
        for end in ['timeout','terminated','uncertainty_cut']:
            e=SkillExecutor('ep',gamma=.5)
            e.begin('wood',torch.zeros(4),threshold=.9,max_steps=5)
            e.advance(ModelStep(torch.ones(4),0.,True))
            tr=e.advance(ModelStep(torch.full((4,),float('nan')) if end=='uncertainty_cut' else torch.ones(4)*2,
                .99,True,**{end:True}))
            self.assertEqual(tr.duration,1 if end=='uncertainty_cut' else 2)
            if end=='uncertainty_cut':
                with self.assertRaises(ValueError): tr.target(torch.tensor(10.))
                self.assertEqual(float(tr.target(torch.tensor(10.),bootstrap_bounds=(0,3))),1.5)
            else:self.assertEqual(float(tr.target(torch.tensor(4.))),0 if end=='terminated' else 1)
        e=SkillExecutor('ep');e.begin('wood',torch.zeros(4),threshold=.9,max_steps=5)
        self.assertIsNone(e.advance(ModelStep(torch.zeros(4),1.,True,uncertainty_cut=True)))
        self.assertFalse(e.active);self.assertEqual(e.ledger.achieved,set())

    def test_invalid_signal_and_timeout_never_advance_success(self):
        e=SkillExecutor('ep')
        e.begin('wood',torch.zeros(4),threshold=.9,max_steps=2)
        with self.assertRaises(ValueError):e.advance(ModelStep(torch.ones(4),float('nan'),True))
        self.assertEqual(e.duration,0)
        e.advance(ModelStep(torch.ones(4),.99,False))
        tr=e.advance(ModelStep(torch.ones(4),.99,False))
        self.assertEqual(tr.reason,'timeout');self.assertEqual(e.ledger.achieved,set())


class BackendTests(unittest.TestCase):
    def make(self,d):
        p,r,c,pm,rm=models(128)
        class World:
            def __init__(self):
                self.tokenizer=torch.nn.Identity().eval();self.dynamics=torch.nn.Identity().eval()
                self.device='cpu';self.resets=0;self.steps=0
                self.action_spec=SimpleNamespace(fingerprint='b'*64,to_tokens=lambda x:x)
            def reset(self,*args,**kwargs):self.resets+=1;self.steps=0
            def step(self,action):
                self.steps+=1
                return None,torch.full((1,1,3,384,640),self.steps/100)
        world=World()
        def encode(pixels):return pixels.mean((0,1,3,4))
        backend=FrozenFeatureBackend(world,p,r,encode,lambda x:x,consistency=lambda *x:True,
            contract=c,policy_metadata=pm,reward_metadata=rm)
        source=Path(d)/'source.txt';source.write_text('synthetic fixture, no real source claim')
        ep=Episode('clip','session','train','offline')
        pixels=torch.zeros(1,1,3,384,640);features=encode(pixels)[None]
        root=OfflineRoot(ep,pixels,torch.zeros(1,1,dtype=torch.long),features,{},torch.zeros(1,2))
        manifest=dict(schema_version=1,episode=vars(ep),frame=0,task_start_frame=0,
            feature_contract_sha256=c.feature_contract_sha256,alignment_evidence_sha256='diagnostic-fixture',
            runtime_contract_sha256=c.fingerprint,sources=[dict(path=str(source),sha256=sha256(source))],
            tensors={k:tensor_hash(getattr(root,k)) for k in ('pixels','frame_actions','features','past_actions')})
        path=Path(d)/'root.json';path.write_text(json.dumps(manifest));root.provenance=dict(manifest_path=str(path),manifest_sha256=sha256(path))
        return backend,root

    def test_backend_contract_reset_and_handoff_preserve_world(self):
        with tempfile.TemporaryDirectory() as d:
            b,root=self.make(d);b.reset(root,0,0);b.step(torch.tensor([0]))
            decision=b.decision_history();h=b.begin_skill(1)
            torch.testing.assert_close(decision,h)
            self.assertEqual(b.world.resets,1);self.assertEqual(b.world.steps,1)
            self.assertIsNone(b.readout.reward_state);self.assertEqual(b.readout.task,0)
            with self.assertRaises(ValueError):b.begin_skill(2)

    def test_explicit_terminal_callback_closes_backend_and_invalid_flags_fail(self):
        with tempfile.TemporaryDirectory() as d:
            b,root=self.make(d);b.outcome=lambda *args:dict(terminated=True,uncertainty=.2)
            b.reset(root,0,0);step=b.step(torch.tensor([0]))
            self.assertTrue(step.terminated);self.assertEqual(step.uncertainty,.2)
            with self.assertRaises(ValueError):b.begin_skill(1)
            b.reset(root,0,0);b.outcome=lambda *args:dict(terminated='yes')
            with self.assertRaises(ValueError):b.step(torch.tensor([0]))
            self.assertFalse(b.ready)

    def test_fixed_collector_calls_real_bridge_without_world_reset(self):
        with tempfile.TemporaryDirectory() as d:
            b,root=self.make(d)
            actor=lambda h,t:[torch.zeros(2)]
            rollout=collect_fixed_skills(b,actor,root,[(0,'wood'),(1,'stone'),(0,'wood')],{0:0.,1:0.},seed=0,
                generator=torch.Generator().manual_seed(0),max_model_calls=10,max_skill_steps=3,gamma=.5)
            self.assertEqual(rollout.model_calls,6);self.assertEqual(b.world.resets,1)
            self.assertEqual([t.reward for t in rollout.transitions],[.5,.5,0.])
            self.assertEqual(rollout.achieved,('stone','wood'))
            self.assertEqual(rollout.stop_reason,'schedule_complete')
            for prev,nxt in zip(rollout.transitions,rollout.transitions[1:]):torch.testing.assert_close(prev.next_history,nxt.history)
            # Fresh root resets the episode ledger, while a budget prevents starting another skill.
            rollout=collect_fixed_skills(b,actor,root,[(0,'wood'),(1,'stone')],{0:0.,1:0.},seed=0,
                generator=torch.Generator().manual_seed(0),max_model_calls=2,max_skill_steps=3)
            self.assertEqual(rollout.model_calls,2);self.assertEqual(len(rollout.transitions),1)
            self.assertEqual(rollout.stop_reason,'budget_exhausted')

if __name__=='__main__': unittest.main()
