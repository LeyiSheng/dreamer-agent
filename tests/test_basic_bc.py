import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from dreamer_agent.learning import HistoryPolicy
from dreamer_agent.export_vpt import export,validate_annotations
from dreamer_agent.train_bc import run as train_bc
from dreamer_agent.preflight_bc import run as preflight
import test_p1_pipeline as pipeline_fixtures


class BasicBCTest(unittest.TestCase):
    def source(self,root):
        p,spec=pipeline_fixtures.PipelineTest().fixture(root)
        m=json.loads(p.read_text())
        m.update(policy_mode="unconditioned",source_reviewed=True,tasks=[])
        m.pop("reviewed")
        for r in m["clips"]:
            r.pop("labels_path");r.pop("labels_sha256")
        p.write_text(json.dumps(m))
        return p,spec

    def export(self,root):
        p,spec=self.source(root)
        with patch("dreamer_agent.export_vpt.decoded_frames",side_effect=lambda path:
            iter([np.full((1,1,4),i+(10 if "000001" in str(path) else 0),dtype=np.float32) for i in range(7)])):
            return export(p,root/"features",spec,lambda f:f.reshape(-1),lambda r:[0,1+r["tick"]%2,3],"b"*64)

    def test_true_unconditioned_policy_has_no_task_parameters(self):
        model=HistoryPolicy(4,2,1,(2,),hidden=8,task_conditioned=False)
        x,a=torch.randn(1,3,4),torch.randn(1,3,2)
        first=model(x,a,None)
        second=model(x,a,torch.full((1,3),999,dtype=torch.long))
        self.assertTrue(torch.equal(first[0],second[0]))
        self.assertIsNone(model.task)
        self.assertFalse(any(name.startswith("task.") for name,_ in model.named_parameters()))

    def test_base_export_and_training_need_no_skill_or_reward_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.export(root)
            m=json.loads(manifest.read_text())
            self.assertEqual(m["task_names"],[])
            self.assertEqual(m["policy_mode"],"unconditioned")
            self.assertFalse((manifest.parent/"reward_labels.json").exists())
            self.assertEqual(list(manifest.parent.glob("labels_*.json")),[])
            check=preflight(manifest,root/"preflight.json",context=4,hidden=8,batch_size=2)
            self.assertEqual(check["optimizer_steps"],0)
            self.assertFalse(check["task_embedding_present"])
            bc=train_bc(manifest,root/"bc",context=4,hidden=8,batch_size=2)
            state=torch.load(bc,weights_only=True)
            self.assertFalse(state["metadata"]["config"]["task_conditioned"])
            self.assertFalse(any(name.startswith("task.") for name in state["model"]))

    def test_basic_mode_cannot_relabel_a_skill_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,_=self.source(Path(tmp));m=json.loads(p.read_text())
            m["tasks"]=["invented_skill"];p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"no skill"):
                validate_annotations(p)

    def test_basic_mode_still_requires_raw_source_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,_=self.source(Path(tmp));m=json.loads(p.read_text())
            m["source_reviewed"]=False;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"raw sources"):
                validate_annotations(p)


if __name__=="__main__":
    unittest.main()
