import unittest
import torch
from dreamer_agent.paper_agent.model import AgentTokenModel,Config,TwoHot
from dreamer_agent.paper_agent.objectives import future_labels,joint_loss

class PaperAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.model=AgentTokenModel(Config(latent_dim=3,image_tokens=2,width=16,heads=2,layers=2,tasks=3,targets=3,reward_bins=31,action_sizes=(2,3)))
        self.z=torch.randn(1,6,2,3);self.actions=torch.randint(0,2,(1,6,2));self.tasks=torch.zeros(1,6,dtype=torch.long)
        self.starts=torch.tensor([[1,0,0,0,0,0]],dtype=torch.bool);self.signal=torch.full((1,6),.5);self.stride=torch.full((1,6),.25)
    def forward(self,z=None,actions=None,tasks=None):
        return self.model(self.z if z is None else z,self.actions if actions is None else actions,self.tasks if tasks is None else tasks,self.starts,self.signal,self.stride)
    def test_task_cannot_change_world_but_can_change_agent(self):
        a=self.forward();b=self.forward(tasks=self.tasks+1)
        torch.testing.assert_close(a['video'],b['video'],rtol=0,atol=0)
        self.assertGreater((a['agent']-b['agent']).abs().max().item(),0)
        a['video'].sum().backward();self.assertEqual(self.model.task.weight.grad.abs().sum().item(),0)
    def test_future_and_current_action_labels_do_not_leak(self):
        a=self.forward();acts=self.actions.clone();acts[:,3:]=1-acts[:,3:]
        b=self.forward(actions=acts)
        for x,y in zip(a['policy'],b['policy']):torch.testing.assert_close(x[:,:4],y[:,:4],rtol=0,atol=0)
        z=self.z.clone();z[:,4:]+=100;b=self.forward(z=z)
        torch.testing.assert_close(a['agent'][:,:4],b['agent'][:,:4],rtol=0,atol=0)
    def test_no_cross_episode_memory(self):
        self.starts[:,3]=True;a=self.forward();z=self.z.clone();z[:,:3]+=100
        acts=self.actions.clone();acts[:,:3]=1-acts[:,:3];b=self.forward(z=z,actions=acts)
        torch.testing.assert_close(a['agent'][:,3:],b['agent'][:,3:],rtol=0,atol=0)
    def test_twohot_roundtrip_and_saturation(self):
        d=TwoHot(255,20).double();v=torch.tensor([-1e10,-1000.,-1.,0.,.3,1.,1000.,1e10],dtype=torch.float64)
        target=d.target(v);torch.testing.assert_close(target.sum(-1),torch.ones_like(v))
        bins=d.support.sign()*torch.expm1(d.support.abs())
        torch.testing.assert_close((target*bins).sum(-1),v.clamp(bins[0],bins[-1]),rtol=1e-10,atol=1e-10)
        self.assertTrue(((target>0).sum(-1)<=2).all())
        with self.assertRaises(ValueError):d.target(torch.tensor([float('nan')]))
    def test_uniform_twohot_is_exactly_zero(self):
        for n in (254,255):
            d=TwoHot(n,20);self.assertEqual(d.mean(torch.zeros(2,n)).abs().sum().item(),0)

    def test_mtp_stops_at_task_switch_even_if_task_returns(self):
        tasks=torch.tensor([[0,1,0,0,0,0]]);valid=torch.ones_like(tasks,dtype=torch.bool)
        _,_,mask=future_labels(self.actions,torch.zeros(1,6),tasks,self.starts,valid,3)
        self.assertFalse(mask[0,0,2]);self.assertFalse(mask[0,5,1]);self.assertTrue(mask[0,2,2])
    def test_mixture_masks(self):
        valid=torch.ones(1,6,dtype=torch.bool);relevant=torch.tensor([[1,1,1,0,0,0]],dtype=torch.bool)
        out=self.forward()
        loss=joint_loss(self.model,out,self.z,self.actions,torch.zeros(1,6),self.tasks,self.starts,valid,relevant,self.signal)
        gv=torch.autograd.grad(loss['video'],out['video'],retain_graph=True)[0];self.assertEqual(gv[:,:3].abs().sum().item(),0)
        gp=torch.autograd.grad(loss['bc'],out['policy'][0],retain_graph=True)[0];self.assertEqual(gp[:,3:].abs().sum().item(),0)
        loss['total'].backward();self.assertGreater(self.model.task.weight.grad.abs().sum().item(),0)
    def test_unknown_reward_is_not_negative(self):
        valid=torch.ones(1,6,dtype=torch.bool);out=self.forward()
        loss=joint_loss(self.model,out,self.z,self.actions,torch.full((1,6),float('nan')),self.tasks,self.starts,valid,valid,self.signal,reward_known=torch.zeros_like(valid))
        self.assertEqual(loss['reward'].item(),0);self.assertTrue(torch.isfinite(loss['total']))
    def test_rl_freezes_everything_except_policy_value(self):
        self.model.freeze_for_rl();before={k:v.clone() for k,v in self.model.named_parameters()}
        out=self.forward();loss=out['policy'][0].square().mean()+self.model.distribution.loss(out['value'],torch.ones(1,6)).mean()
        opt=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.001);loss.backward();opt.step()
        changed=[]
        for name,p in self.model.named_parameters():
            if not torch.equal(p,before[name]):changed.append(name)
            if not name.startswith(('policy.','value.')):self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,before[name]))
        self.assertTrue(any(k.startswith('policy.') for k in changed));self.assertTrue(any(k.startswith('value.') for k in changed))

if __name__=='__main__':unittest.main()
