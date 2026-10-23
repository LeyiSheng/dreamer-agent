"""Build reviewed observation-transition labels from the P1b event audit."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from .provenance import sha256
def comparable(a,b):
 x,y=a.get("stats"),b.get("stats")
 if not isinstance(x,dict) or not isinstance(y,dict) or not x or not y:return False
 if any(type(v)is not int or v<0 for v in list(x.values())+list(y.values())):return False
 if b.get("tick")!=a.get("tick",-2)+1 or b.get("serverTick",0)<a.get("serverTick",0):return False
 if set(x)-set(y):return False
 return not any(y[k]<v for k,v in x.items() if not k.startswith("minecraft.custom:"))
def clip_labels(rows,events,start,stop,task,task_id):
 by_frame=defaultdict(list)
 for e in events:
  if e["task"]==task:by_frame[e["frame"]].append(e)
 result=[];complete=0
 for i in range(start,stop):
  event=None
  if i+1<stop and comparable(rows[i],rows[i+1]):
   es=by_frame[i+1]
   event=1 if any(e["accepted"] for e in es) else None if es else 0
  if i+1>=stop:completion=None
  else:
   if event==1:complete=1
   elif event is None and complete!=1:complete=None
   completion=complete
  result.append(dict(task=task_id,event=event,completion=completion))
 return result
def main():
 ap=argparse.ArgumentParser()
 for k in ("manifest","coverage","sources","output"):ap.add_argument("--"+k,type=Path,required=True)
 a=ap.parse_args();m=json.loads(a.manifest.read_text());cov=json.loads(a.coverage.read_text())
 review_path=a.manifest.parent/"event_review.json";review={r["episode_id"]:r["events"] for r in json.loads(review_path.read_text())}
 assert m["coverage_sha256"]==sha256(a.coverage)
 sources=json.loads(a.sources.read_text());assert cov["sources_sha256"]==sha256(a.sources)
 source={r["episode_id"]:r for r in sources["clips"]}
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 cache={};records=[];counts={split:{task:{h:[0,0,0] for h in ("event","completion")} for task in m["task_names"]} for split in ("train","calibration")}
 for ix,r in enumerate(m["episodes"]):
  assert r["split"] in ("train","calibration")
  parent=r["parent_episode_id"]
  if parent not in cache:
   src=source[parent];p=Path(src["actions_path"]);assert sha256(p)==src["actions_sha256"]==r["source_actions_sha256"]
   assert src["split"]==r["split"] and src["source_episode_id"]==r["source_episode_id"]
   cache={parent:[json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]}
  task=r["event_evidence"]["task"];tid=m["task_names"].index(task)
  labels=clip_labels(cache[parent],review[parent],r["start"],r["stop"],task,tid)
  p=out/f"labels_{ix:05d}.json";p.write_text(json.dumps(labels))
  records.append(dict(episode_id=r["episode_id"],path=p.name,sha256=sha256(p)))
  # Count exactly the transitions consumed by current 32-frame BC history protocol.
  for i,row in enumerate(labels):
   if i==len(labels)-1 or i%32==31:continue
   for h in ("event","completion"):counts[r["split"]][task][h][2 if row[h] is None else row[h]]+=1
 result=dict(schema_version=1,origin="offline",reviewed=True,reviewer="automated cumulative-counter and P1b event-evidence protocol v1; not manual video review",
  feature_manifest_sha256=sha256(a.manifest),event_review_sha256=sha256(review_path),coverage_sha256=sha256(a.coverage),code_sha256=sha256(__file__),
  label_protocol="observed_transition_event_and_since_clip_completion_v1",episodes=records,
  semantics={"event":"row t labels counter increment observed in stats[t+1]-stats[t]; no physical latency correction",
   "completion":"at least one supported task event since this clip start, evaluated at t+1; absorbing within clip, not environment termination"},
  limitations=["Task-selected clips; general background false positives not calibrated.","Initial positive cumulative history is not success in this attempt.","Context boundary transitions excluded by trainer.","No true-state policy input."])
 (out/"reward_labels.json").write_text(json.dumps(result,indent=2))
 (out/"coverage.json").write_text(json.dumps(counts,indent=2))
 for split in counts:
  for task,heads in counts[split].items():
   print(split,task,heads,flush=True)
 if any(min(v[:2])==0 for hs in counts["train"].values() for v in hs.values()):raise ValueError("Missing positive/negative support")
if __name__=="__main__":main()
