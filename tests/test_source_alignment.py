import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from dreamer_agent.sources import session_id,session_split,screen,require_export_ready
from dreamer_agent.alignment_diagnostic import lag_scores


class SourceAndAlignmentTest(unittest.TestCase):
    def test_original_session_survives_segment_timestamp(self):
        a=session_id("player-one-abcdef012345-20220101-010000")
        b=session_id("player-one-abcdef012345-20220101-010500")
        self.assertEqual(a,b)
        self.assertEqual(session_split(a,17),session_split(b,17))

    def test_unknown_filename_not_silently_new_session(self):
        with self.assertRaises(ValueError):
            session_id("demo_000")

    def test_bad_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"player-one-abcdef012345-20220101-010000.jsonl"
            p.write_text("{}")
            p.with_suffix(".mp4").write_bytes(b"broken")
            result=SimpleNamespace(returncode=1,stderr="moov atom not found")
            with patch("dreamer_agent.sources.subprocess.run",return_value=result):
                record=screen(p,root)
            self.assertEqual(record["status"],"rejected")
            self.assertIn("moov",record["reason"])
            with self.assertRaises(ValueError):
                require_export_ready(record)

    def test_unknown_frame_rate_is_rejected_without_aborting_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"player-one-abcdef012345-20220101-010000.jsonl"
            p.write_text("{}")
            p.with_suffix(".mp4").write_bytes(b"fixture")
            result=SimpleNamespace(returncode=0,stdout='{"streams":[{"avg_frame_rate":"0/0","nb_frames":"21"}]}')
            with patch("dreamer_agent.sources.subprocess.run",return_value=result):
                record=screen(p,root)
            self.assertEqual(record["status"],"rejected")
            self.assertFalse(record["export_ready"])

    def test_readable_still_not_export_ready(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"player-one-abcdef012345-20220101-010000.jsonl"
            p.write_text("{}")
            p.with_suffix(".mp4").write_bytes(b"fixture")
            result=SimpleNamespace(returncode=0,stdout=json.dumps({"streams":[{"avg_frame_rate":"20/1","nb_frames":"21"}]}))
            with patch("dreamer_agent.sources.subprocess.run",return_value=result):
                record=screen(p,root)
            self.assertEqual(record["status"],"alignment_pending")
            with self.assertRaises(ValueError):
                require_export_ready(record)

    def test_known_lag_recovered(self):
        actions=np.random.default_rng(5).normal(size=150)
        motion=actions[1:].copy()
        result=lag_scores(motion,actions,np.ones(len(motion),dtype=bool))
        best=max(result,key=lambda r:r["correlation"])
        self.assertEqual(best["lag"],1)
        self.assertAlmostEqual(best["correlation"],1.)
        self.assertEqual(len({r["n"] for r in result}),1)

    def test_constant_motion_has_no_alignment_claim(self):
        result=lag_scores(np.zeros(50),np.ones(50),np.ones(50,dtype=bool))
        self.assertTrue(all(r["correlation"] is None for r in result))

    def test_nan_mask_is_common_to_all_lags(self):
        values=np.arange(80,dtype=float)
        values[30]=np.nan
        result=lag_scores(np.arange(79),values,np.ones(79,dtype=bool))
        self.assertEqual(len({r["n"] for r in result}),1)
        self.assertTrue(all(r["correlation"] is not None for r in result))


if __name__=="__main__":
    unittest.main()
