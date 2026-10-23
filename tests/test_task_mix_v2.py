import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dreamer_agent.task_mix_data import select_sources, validate_splits, objective_clips, shuffled_batches
from dreamer_agent.task_mix_windows import windows
from dreamer_agent.train_task_mix_v2 import weighted_loss,compatible


def ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    model=torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():model.weight.copy_(torch.arange(6).reshape(3,2)/10)
    model=DDP(model)
    # Rank 1 has only a padding batch. Correct global gradient must remain unchanged.
    x=torch.tensor([[[1.,2.],[3.,4.]]])
    target=torch.tensor([[[[1]],[[2]]]])
    valid=torch.tensor([[[True],[False]]]) if rank==0 else torch.zeros(1,2,1,dtype=torch.bool)
    logits=[model(x).unsqueeze(-2)]
    num,mass=weighted_loss(logits,target,valid,torch.zeros(1,2,dtype=torch.long),torch.ones(1))
    dist.all_reduce(mass)
    (num*2/mass).backward()
    torch.save(model.module.weight.grad,output+str(rank))
    dist.destroy_process_group()


class TaskMixTests(unittest.TestCase):
    def records(self):
        return [dict(stem=str(i),split='train',status='eligible_pending_full_decode',version=str(i%2),identity=str(i%4),source_session='session-'+str(i),actions_sha256=str(i),frames=513,clips=[dict(start=0,stop=10,task='rare' if i==19 else 'common',background=False)]) for i in range(20)]

    def test_subset_unique_nested_and_rare(self):
        rows=self.records();a=select_sources(rows,.2,0,['common','rare'])
        self.assertEqual(len(a),4);self.assertIn('19',{r['stem'] for r in a})
        self.assertEqual(a,select_sources(list(reversed(rows)),.2,0,['common','rare']))
        self.assertTrue({r['stem'] for r in a}<={r['stem'] for r in select_sources(rows,1,0,['common','rare'])})

    def test_no_heldout_selection(self):
        rows=self.records();rows[-1]['split']='test'
        self.assertNotIn('19',{r['stem'] for r in select_sources(rows,1,0,['common','rare'])})

    def test_split_leakage(self):
        rows=self.records();validate_splits(rows)
        rows[-1]['split']='test';rows[-1]['source_session']=rows[0]['source_session']
        with self.assertRaises(ValueError):validate_splits(rows)
        rows[-1]['source_session']='new';rows[-1]['actions_sha256']=rows[0]['actions_sha256']
        with self.assertRaises(ValueError):validate_splits(rows)

    def test_general_covers_every_frame_without_fake_goal(self):
        r=self.records()[0];clips=objective_clips(r,['common','rare'])
        covered=np.zeros(r['frames'],dtype=int)
        for c in clips:
            if c['stream']=='general':
                self.assertEqual(c['task_id'],2);covered[c['start']:c['stop']]+=1
        np.testing.assert_array_equal(covered,np.ones(513))

    def test_context_targets_and_past_never_cross_clip(self):
        w=windows(np.ones((5,2)),np.array([[0],[1],[2],[1],[0]]),context=3,targets=3,action_sizes=[3],task_id=1)
        self.assertEqual(w[0]['valid'].sum(),9)
        self.assertEqual(w[1]['valid'].sum(),3)
        np.testing.assert_array_equal(w[0]['past_actions'][0],np.zeros(3))
        self.assertEqual(w[1]['past_actions'][0].argmax(),2)
        self.assertFalse(w[1]['valid'][2].any())

    def test_buffer_no_loss_or_duplication_and_replay(self):
        def batches():return list(shuffled_batches(iter(range(43)),8,16,5))
        a=batches();self.assertEqual(a,batches())
        self.assertEqual(sorted(x for b in a for x in b),list(range(43)))
        self.assertTrue(all(len(b)==8 for b in a[:-1]))
        replay=iter(shuffled_batches(iter(range(43)),8,16,5));next(replay);next(replay)
        self.assertEqual(list(replay),a[2:])

    def test_ddp_gradient_matches_single_global_batch_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(ddp_worker,args=(str(Path(d)/'rendezvous'),str(Path(d)/'grad')),nprocs=2,join=True)
            model=torch.nn.Linear(2,3,bias=False)
            with torch.no_grad():model.weight.copy_(torch.arange(6).reshape(3,2)/10)
            torch.nn.functional.cross_entropy(model(torch.tensor([[1.,2.]])),torch.tensor([1])).backward()
            for rank in range(2):torch.testing.assert_close(torch.load(str(Path(d)/'grad')+str(rank),weights_only=True),model.weight.grad)

    def test_masked_task_weight_loss(self):
        x=torch.tensor([[[[2.,0.]],[[0.,2.]],[[99.,0.]]]],requires_grad=True)
        y=torch.tensor([[[[0]],[[0]],[[1]]]])
        n,d=weighted_loss([x],y,torch.tensor([[[1],[1],[0]]],dtype=torch.bool),torch.tensor([[0,1,1]]),torch.tensor([1.,3.]))
        expected=(torch.nn.functional.cross_entropy(x[0,0],torch.tensor([0]))+3*torch.nn.functional.cross_entropy(x[0,1],torch.tensor([0])))/4
        torch.testing.assert_close(n/d,expected);(n/d).backward();self.assertEqual(x.grad[0,2].abs().sum(),0)

    def test_checkpoint_contract_rejects_changed_features(self):
        md=dict(config={},tasks=['a'],action_spec_sha256='a',extractor_sha256='b',context=2,audit_sha256='c')
        compatible(md,md)
        with self.assertRaises(ValueError):compatible(md,dict(md,extractor_sha256='changed'))

if __name__=='__main__':unittest.main()
