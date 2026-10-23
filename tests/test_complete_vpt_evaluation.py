import unittest
from dreamer_agent.complete_vpt_evaluation import completed_records,expected_jobs
class ResumeEvalTests(unittest.TestCase):
 def test_remaining_exactly24(self):
  jobs=expected_jobs({'world_seeds':list(range(21000,21010))})
  records=[dict(status='ok',arm=a,world_seed=s) for a,s in jobs[:16]]
  records.append(dict(status='failed',arm=jobs[16][0],world_seed=jobs[16][1]))
  completed=completed_records(records,jobs)
  remaining=[k for k in jobs if k not in completed]
  self.assertEqual(len(remaining),24);self.assertEqual(remaining[0],jobs[16])
 def test_duplicate_or_foreign_success_rejected(self):
  row=dict(status='ok',arm='baseline',world_seed=1)
  with self.assertRaises(ValueError):completed_records([row,row],[('baseline',1)])
  with self.assertRaises(ValueError):completed_records([row],[('baseline',2)])
 def test_retry_failure_not_counted_as_success(self):
  rows=[dict(status=s,arm='rl_0',world_seed=1) for s in ['failed','failed','ok']]
  self.assertEqual(len(completed_records(rows,[('rl_0',1)])),1)
 def test_schedule_covers_each_arm_once(self):
  jobs=expected_jobs({'world_seeds':[1,2]});self.assertEqual(len(jobs),8);self.assertEqual(len(set(jobs)),8)
if __name__=='__main__':unittest.main()
