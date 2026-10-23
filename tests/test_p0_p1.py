import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from dreamer_agent.video_audit import inspect_pair
import numpy as np
from dreamer_agent.actions import ActionSpec
from dreamer_agent.calibration import metrics, select_threshold, wilson
from dreamer_agent.dataset import OfflineWindows
from dreamer_agent.provenance import sha256, write_report
from dreamer_agent.vpt_audit import inspect
from dreamer_agent.train_bc import run


class P0P1Test(unittest.TestCase):
    def fixture(self, root):
        # Synthetic contract fixture, never passed off as actual offline evidence.
        spec = ActionSpec(((7,), (10, 20), (30, 40, 50), (8,)), "a"*64, 20., 1)
        records = []
        for split, session in [("train", "one"), ("calibration", "two")]:
            path = root / (split + ".npz")
            features = np.arange(20, dtype=np.float32).reshape(5, 4)
            if split == "calibration":
                features += 1
            np.savez(path, features=features, actions=np.zeros((5, 2), dtype=np.int64),
                     tasks=np.array([0, 0, 1, 1, 1]), frames=np.arange(5))
            records.append(dict(path=path.name, sha256=sha256(path), episode_id=split,
                                source_episode_id=session, split=split, origin="offline"))
        manifest = dict(schema_version=1, origin="offline", action_spec=spec.to_dict(),
                        feature_spec=dict(kind="frozen_visual", frozen=True, dimension=4,
                                          extractor_sha256="b"*64, preprocessing_sha256="c"*64),
                        tasks=2, episodes=records)
        p = root / "manifest.json"
        p.write_text(json.dumps(manifest))
        return p, manifest

    def test_video_extra_frame_is_not_silently_trimmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, actions = root/"clip.mp4", root/"actions.jsonl"
            video.write_bytes(b"synthetic fixture")
            actions.write_text(json.dumps(dict(tick=0, keyboard=dict(keys=[]), mouse=dict(dx=0, dy=0, buttons=[]))))
            result = SimpleNamespace(stdout=json.dumps(dict(streams=[dict(avg_frame_rate="20/1", nb_frames="2")])))
            with patch("dreamer_agent.video_audit.subprocess.run", return_value=result):
                report = inspect_pair(video, actions, 20)
            self.assertFalse(report["length_and_rate_match"])
            self.assertTrue(report["possible_extra_observation"])
            self.assertFalse(report["action_observation_offset_verified"])

    def test_nonfinite_mouse_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.jsonl"
            p.write_text(json.dumps(dict(tick=0, keyboard=dict(keys=[]), mouse=dict(dx=float("nan"), dy=0, buttons=[]))))
            self.assertFalse(inspect(p)["eligible_for_alignment_review"])

    def test_constant_only_action_space_rejected(self):
        with self.assertRaises(ValueError):
            ActionSpec(((7,), (8,)), "a"*64, 20, 1)

    def test_action_roundtrip_and_fixed_slots(self):
        s = ActionSpec(((7,), (4, 5), (8,)), "a"*64, 20., 1)
        actions = np.array([[0], [1]])
        np.testing.assert_array_equal(s.to_tokens(actions), [[7, 4, 8], [7, 5, 8]])
        np.testing.assert_array_equal(s.to_local(s.to_tokens(actions)), actions)
        with self.assertRaises(ValueError):
            s.to_local(np.array([[7, 4, 4]]))

    def test_illegal_local_action_rejected(self):
        s = ActionSpec(((1, 2),), "a"*64, 20, 1)
        for action in [np.array([[2]]), np.array([[-1]]), np.array([[0.5]])]:
            with self.assertRaises(ValueError):
                s.to_tokens(action)

    def test_mtp_ends_at_task_switch_and_episode_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _ = self.fixture(Path(tmp))
            data = OfflineWindows(p, "train", context=4, targets=3)
            row = data[0]
            self.assertEqual(row["valid"].tolist(), [[True, True, False], [True, False, False],
                                                    [True, True, True], [True, True, False]])
            last = data[1]
            self.assertEqual(last["valid"].sum().item(), 1)
            self.assertEqual(row["past_actions"][0].sum().item(), 0)
            self.assertEqual(row["past_actions"][1].sum().item(), 2)

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, _ = self.fixture(root)
            with (root/"train.npz").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                OfflineWindows(p, "train")

    def test_cross_session_leak_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, m = self.fixture(Path(tmp))
            m["episodes"][1]["source_episode_id"] = "one"
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "crosses"):
                OfflineWindows(p, "train")

    def test_privileged_feature_array_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, m = self.fixture(root)
            path = root/"train.npz"
            with np.load(path) as src:
                arrays = dict(src)
            np.savez(path, **arrays, inventory=np.ones(5))
            m["episodes"][0]["sha256"] = sha256(path)
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "privileged"):
                OfflineWindows(p, "train")

    def test_raw_missing_deltas_not_assumed_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.jsonl"
            p.write_text(json.dumps(dict(tick=0, keyboard=dict(keys=[]), mouse=dict(buttons=[])))+"\n")
            self.assertFalse(inspect(p)["eligible_for_alignment_review"])

    def test_raw_tick_gaps_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.jsonl"
            p.write_text("\n".join(json.dumps(dict(tick=t, keyboard=dict(keys=[]),
                         mouse=dict(dx=0, dy=0, buttons=[]))) for t in [0, 2]))
            self.assertEqual(inspect(p)["tick_gaps"], 1)
            self.assertFalse(inspect(p)["eligible_for_alignment_review"])

    def test_no_predictions_not_perfect_precision(self):
        r = metrics([.1, .2], [0, 1], .9)
        self.assertIsNone(r["precision"])
        self.assertIsNone(r["precision_interval"])

    def test_small_perfect_set_cannot_unlock_reward(self):
        self.assertFalse(select_threshold([.99]*5, [1]*5)["passed"])

    def test_precision_evidence(self):
        scores, labels = [.99]*100 + [.01]*100, [1]*100 + [0]*100
        r = select_threshold(scores, labels)
        self.assertTrue(r["passed"])
        self.assertEqual(r["fp"], 0)
        self.assertGreater(r["precision_interval"][0], .95)

    def test_wilson_reference(self):
        lo, hi = wilson(7, 1000)
        self.assertAlmostEqual(lo, .003395, places=5)
        self.assertAlmostEqual(hi, .014379, places=5)

    def test_report_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"report.json"
            write_report(p, dict(result=1))
            with self.assertRaises(FileExistsError):
                write_report(p, dict(result=2))

    def test_bc_training_contract_cpu_only_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, _ = self.fixture(root)
            output = root/"run"
            checkpoint = run(p, output, context=4, targets=3, hidden=8, batch_size=2)
            self.assertTrue(checkpoint.exists())
            row = json.loads((output/"metrics.jsonl").read_text())
            self.assertFalse(row["task_success_measured"])
            self.assertTrue(np.isfinite(row["train_nll"]))
            with self.assertRaises(FileExistsError):
                run(p, output)


if __name__ == "__main__":
    unittest.main()
