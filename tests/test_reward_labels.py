import unittest
from dreamer_agent.prepare_reward_labels import clip_labels
K="minecraft.pickup:minecraft.oak_log"
def row(t,v):return dict(tick=t,serverTick=t,stats={K:v})
class RewardLabelTests(unittest.TestCase):
 def test_next_observation_event_not_tail_label(self):
  r=clip_labels([row(0,5),row(1,5),row(2,6)],[dict(frame=2,task="wood",accepted=True)],0,3,"wood",0)
  self.assertEqual([x["event"] for x in r],[0,1,None])
  self.assertEqual([x["completion"] for x in r],[0,1,None])
 def test_cumulative_history_is_not_completion(self):
  r=clip_labels([row(0,8),row(1,8)],[],0,2,"wood",0)
  self.assertEqual(r[0]["completion"],0)
 def test_completion_persists_event_does_not(self):
  r=clip_labels([row(i,v) for i,v in enumerate([0,1,1,1])],[dict(frame=1,task="wood",accepted=True)],0,4,"wood",0)
  self.assertEqual([x["event"] for x in r],[1,0,0,None])
  self.assertEqual([x["completion"] for x in r],[1,1,1,None])
 def test_unconfirmed_event_unknown_not_negative(self):
  r=clip_labels([row(0,0),row(1,1),row(2,1)],[dict(frame=1,task="wood",accepted=False)],0,3,"wood",0)
  self.assertIsNone(r[0]["event"]);self.assertIsNone(r[1]["completion"])
 def test_counter_reset_is_unknown(self):
  r=clip_labels([row(0,10),row(1,1)],[],0,2,"wood",0)
  self.assertIsNone(r[0]["event"])
 def test_other_task_does_not_complete(self):
  r=clip_labels([row(0,0),row(1,0)],[dict(frame=1,task="stone",accepted=True)],0,2,"wood",0)
  self.assertEqual(r[0]["completion"],0)
if __name__=="__main__":unittest.main()
