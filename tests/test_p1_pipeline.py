"""Synthetic software integration; never evidence of Minecraft success."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from dreamer_agent.actions import ActionSpec
from dreamer_agent.provenance import sha256
from dreamer_agent.export_vpt import export, validate_annotations, PROTOCOL
from dreamer_agent.dataset import OfflineWindows
from dreamer_agent.train_bc import run as train_bc
from dreamer_agent.train_reward import run as train_reward, load_labels, calibrate, calibration_with_confirmation


class PipelineTest(unittest.TestCase):
    def fixture(self, root):
        spec=ActionSpec(((0,),(1,2),(3,)), "a"*64,20,1)
        clips=[]
        for index,split in enumerate(("train","calibration")):
            stem=f"player-{index:012x}-20220901-120000"
            video=root/(stem+".mp4");video.write_bytes(b"synthetic video "+bytes([index]))
            actions=root/(stem+".jsonl")
            rows=[dict(tick=t,keyboard=dict(keys=[]),mouse=dict(dx=index,dy=0,buttons=[])) for t in range(6)]
            actions.write_text("\n".join(json.dumps(r) for r in rows))
            labels=root/(stem+".labels.json")
            labels.write_text(json.dumps([dict(task=0,event=t%2,completion=1-t%2) for t in range(6)]))
            r=dict(episode_id=split,source_episode_id=f"player-{index:012x}",split=split,origin="offline",start=0,stop=6)
            for kind,p in (("video",video),("actions",actions),("labels",labels)):
                r[kind+"_path"]=str(p);r[kind+"_sha256"]=sha256(p)
            clips.append(r)
        p=root/"annotations.json"
        p.write_text(json.dumps(dict(schema_version=1,origin="offline",reviewed=True,reviewer="synthetic unittest",
            pairing_protocol=PROTOCOL,tasks=["fixture"],clips=clips)))
        return p,spec

    def do_export(self,root,extra=1):
        p,spec=self.fixture(root)
        # Patches represent synthetic fixtures, not real visual encodings.
        with patch("dreamer_agent.export_vpt.decoded_frames",
                   side_effect=lambda path: iter([np.full((1,1,4),i+(10 if "000001" in str(path) else 0),dtype=np.float32) for i in range(6+extra)])):
            return export(p,root/"features",spec,lambda frame:frame.reshape(-1),lambda row:[0,1+row["tick"]%2,3],"b"*64)

    def test_full_export_bc_reward_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.do_export(root)
            data=OfflineWindows(manifest,"train",4,2)
            self.assertEqual(data.episodes[0]["frames"].tolist(),list(range(6)))
            m=json.loads(manifest.read_text())
            self.assertEqual(m["episodes"][0]["unused_tail_frames"],1)
            bc=train_bc(manifest,root/"bc",context=4,targets=2,hidden=8,batch_size=2)
            before=sha256(bc)
            reward=train_reward(manifest,manifest.parent/"reward_labels.json",bc,root/"reward",hidden=8)
            self.assertEqual(before,sha256(bc))
            self.assertTrue(reward.exists())
            cal=json.loads((root/"reward/calibration.json").read_text())
            self.assertFalse(cal["rl_ready"])
            self.assertTrue((root/"reward/calibration_scores.json").exists())

    def test_bad_frame_count_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,"T or T"):
                self.do_export(root,extra=2)
            self.assertFalse((root/"features/manifest.json").exists())

    def test_unreviewed_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_=self.fixture(root)
            m=json.loads(p.read_text());m["reviewed"]=False;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"Reviewed"):
                validate_annotations(p)

    def test_raw_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_=self.fixture(root)
            m=json.loads(p.read_text());Path(m["clips"][0]["video_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"hash"):
                validate_annotations(p)

    def test_unknown_task_not_defaulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_=self.fixture(root)
            m=json.loads(p.read_text());r=m["clips"][0];lp=Path(r["labels_path"])
            rows=json.loads(lp.read_text());rows[0]["task"]=None;lp.write_text(json.dumps(rows))
            r["labels_sha256"]=sha256(lp);p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"task"):
                validate_annotations(p)

    def test_overlapping_clips_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_=self.fixture(root)
            m=json.loads(p.read_text());r=dict(m["clips"][0]);r["episode_id"]="overlap";m["clips"].append(r)
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"Overlapping"):
                validate_annotations(p)

    def test_label_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.do_export(root)
            labels=manifest.parent/"reward_labels.json"
            m=json.loads(labels.read_text());m["feature_manifest_sha256"]="0"*64
            labels.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"different feature"):
                load_labels(labels,manifest)

    def test_confirmation_never_changes_selected_threshold(self):
        selected=[dict(task=0,head="event",label=int(i<100),score=.9 if i<100 else .1,session="a") for i in range(200)]
        # An overconfident wrong reward head must fail independent confirmation.
        heldout=[dict(task=0,head="event",label=int(i<100),score=.1 if i<100 else .99,session="b") for i in range(200)]
        r=calibration_with_confirmation(selected+heldout,1,["a","b"])
        head=r["tasks"][0]["heads"]["event"]
        self.assertEqual(head["selection"]["threshold"],.9)
        self.assertFalse(head["confirmation"]["passed"])
        self.assertEqual(head["confirmation"]["metrics"]["fp"],100)
        self.assertFalse(r["rl_ready"])

    def test_calibration_all_positive_cannot_enable_head(self):
        rows=[dict(task=0,head="event",label=1,score=.99,session="one") for _ in range(100)]
        r=calibrate(rows,1)
        self.assertFalse(r["tasks"][0]["heads"]["event"]["selection"]["passed"])
        self.assertFalse(r["rl_ready"])


if __name__=="__main__":
    unittest.main()
