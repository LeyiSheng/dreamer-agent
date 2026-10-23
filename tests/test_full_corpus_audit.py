import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from dreamer_agent.audit_full_corpus import audit
class FullAuditTests(unittest.TestCase):
 def test_test_records_never_read(self):
  self.assertEqual(audit(dict(split="test",actions_path="does_not_exist"))["status"],"held_out_not_read")
 def test_baseline_counter_not_positive_and_background_retained(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"a.jsonl";rows=[dict(tick=i,hotbar=0,mouse=dict(dx=0,dy=0,buttons=[]),keyboard=dict(keys=[]),stats={"minecraft.pickup:minecraft.oak_log":5}) for i in range(20)]
   p.write_text("\n".join(json.dumps(r) for r in rows))
   ff=SimpleNamespace(returncode=0,stdout=json.dumps(dict(streams=[dict(width=640,height=360,avg_frame_rate="20/1",nb_frames="20")])))
   with patch("dreamer_agent.audit_full_corpus.subprocess.run",return_value=ff):
    r=audit(dict(split="train",actions_path=str(p),video_path="a.mp4",stem="fixture"))
   self.assertEqual(r["status"],"eligible_pending_full_decode");self.assertEqual(r["events"],[])
   self.assertTrue(r["clips"][0]["background"])
 def test_missing_hotbar_quarantined_not_assumed(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"a.jsonl";p.write_text('{"tick":0}\n{"tick":1}')
   r=audit(dict(split="train",actions_path=str(p)));self.assertEqual(r["status"],"needs_review")
if __name__=="__main__":unittest.main()
