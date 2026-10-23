import json
from pathlib import Path
import tempfile
import unittest

from dreamer_agent.prepare_budget_roots import candidates
from dreamer_agent.vpt_budget_rl import load_budget, next_seed, validate_config
from dreamer_agent.vpt_direct_rl import root_index


class BudgetTests(unittest.TestCase):
    def test_resume_does_not_restart_eight_hours(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'budget.json'
            first=load_budget(p,28800,1000000.)
            recovered=load_budget(p,28800,1000000.+10000.)
            self.assertEqual(first,recovered)
            self.assertEqual(recovered['deadline_timestamp']-(1000000.+10000.),18800)
            with self.assertRaises(ValueError):load_budget(p,57600,1000000.)

    def test_fair_seed_interleaving_including_partial_checkpoint_recovery(self):
        versions={0:20,1:0,2:20}
        sequence=[]
        for _ in range(26):
            seed=next_seed(versions);sequence.append(seed);versions[seed]+=1
        self.assertEqual(sequence[:20],[1]*20)
        self.assertEqual(versions,{0:22,1:22,2:22})

    def test_each_seed_covers_all_512_roots_before_repetition(self):
        for seed in (0,1,2):
            for cycle in (0,1):
                indices=[root_index(i,seed,512) for i in range(cycle*512,(cycle+1)*512)]
                self.assertEqual(set(indices),set(range(512)))

    def test_protocol_rejects_silent_budget_or_objective_expansion(self):
        cfg=dict(scope='vpt_proxy_rl_time_budget_v2',user_authorized=True,budget_seconds=28800,
            seeds=[0,1,2],horizon=16,bc_weight=0,learning_rate=3e-5,kl_limit=1.,checkpoint_every=20)
        validate_config(cfg)
        for k,v in [('budget_seconds',57600),('horizon',64),('bc_weight',1),('seeds',[1]),('user_authorized',False)]:
            with self.assertRaises(ValueError):validate_config(dict(cfg,**{k:v}))

    def test_candidates_exclude_heldout_session_aliases_and_are_reproducible(self):
        rows=[]
        for i in range(12):
            rows.append(dict(stem=f'clip{i}',version=f'{6+i%3}xx',identity=f'alias{i}',
                source_session=f'alias{i}-session{i}',split='train',status='eligible_pending_full_decode',
                frames=100,video_path=f'video{i}',actions_path=f'actions{i}',actions_sha256=str(i),
                events=[dict(task='mine_log',accepted=True,frame=60)]))
        # A differently named source in test with the same conservative session key excludes train alias.
        rows.append(dict(rows[0],stem='heldout0',source_session='other-session0',split='test'))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'audit.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
            a=candidates(path,123);b=candidates(path,123)
            self.assertEqual(a,b);self.assertEqual(len(a),11)
            self.assertNotIn('session0',{r['group'] for r in a})
            self.assertEqual(len({r['version'] for r in a[:3]}),3)


if __name__ == '__main__':unittest.main()
