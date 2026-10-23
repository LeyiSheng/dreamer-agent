import unittest,json,tempfile
from pathlib import Path
import torch
import test_basic_bc as fixtures
from dreamer_agent.train_bc import run
from dreamer_agent.provenance import sha256
from dreamer_agent.prepare_task_bc import ranges,inventory_count
class TaskBCTest(unittest.TestCase):
 def test_ambiguous_goal_prefix_removed(self):
  es=[dict(frame=30,task="a"),dict(frame=70,task="b")]
  rr=ranges(es,100)
  self.assertEqual([(a,b) for a,b,_ in rr],[(0,31),(31,71)])
 def test_no_overlapping_dense_events(self):
  rr=ranges([dict(frame=i,task="a") for i in [30,31,32,50]],100)
  self.assertEqual([(a,b) for a,b,_ in rr],[(0,31),(31,51)])
 def test_stat_inventory_item_match(self):
  item="minecraft.craft_item:minecraft.wooden_pickaxe".split(":")[-1].removeprefix("minecraft.")
  self.assertEqual(inventory_count({"inventory":[{"type":"wooden_pickaxe","quantity":1}]},item),1)
 def test_inventory_missing_unknown(self):
  self.assertIsNone(inventory_count({},"oak_log"))
 def test_basic_initialization_and_wrong_source_rejection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);manifest=fixtures.BasicBCTest().export(root)
   bc=run(manifest,root/"base",context=4,hidden=8,batch_size=2)
   m=json.loads(manifest.read_text());m.update(tasks=2,task_names=["a","b"],policy_mode="task_conditioned",parent_manifest_sha256=sha256(manifest))
   target=manifest.parent/"task_manifest.json";target.write_text(json.dumps(m))
   cp=run(target,root/"task",context=4,hidden=8,batch_size=2,init_checkpoint=bc)
   state=torch.load(cp,weights_only=True)
   self.assertEqual(state["metadata"]["initialization"]["sha256"],sha256(bc))
   self.assertIn("task.weight",state["model"])
   m["parent_manifest_sha256"]="0"*64;target.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,"source manifest"):
    run(target,root/"bad",context=4,hidden=8,batch_size=2,init_checkpoint=bc)
if __name__=="__main__":unittest.main()
