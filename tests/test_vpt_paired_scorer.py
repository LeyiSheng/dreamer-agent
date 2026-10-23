import unittest
from dreamer_agent.eval_vpt_rl_paired import inventory_counts,milestone_counts
class ScorerTests(unittest.TestCase):
 def test_exact_items_not_lookalikes(self):
  got=milestone_counts(inventory_counts({'inventory':{'minecraft:oak_log':2,'log2':1,'diamond_ore':5,'diamond_pickaxe':1,'diamond':0,'wooden_pickaxe':1}}))
  self.assertEqual(got['log'],3);self.assertEqual(got['diamond'],0);self.assertEqual(got['wooden_pickaxe'],1)
 def test_inventory_required(self):
  with self.assertRaises(ValueError):inventory_counts({'pov':None})
if __name__=='__main__':unittest.main()
