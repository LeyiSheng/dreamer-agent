"""Build bounded retrospective task clips using audited offline event evidence."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from .provenance import sha256
def inventory_count(row,item):
 inv=row.get("inventory")
 if not isinstance(inv,list):return None
 if any(not isinstance(x,dict) or not isinstance(x.get("type"),str) or type(x.get("quantity")) is not int for x in inv):return None
 return sum(x["quantity"] for x in inv if x["type"].removeprefix("minecraft:")==item)
def ranges(events,length,lookback=128):
 selected=[];end=0
 for e in sorted(events,key=lambda e:e["frame"]):
  stop=e["frame"]+1;start=max(end,stop-lookback)
  if stop-start<16:continue
  # Another supported task in the lookback makes the preceding goal ambiguous.
  other=[x["frame"]+1 for x in events if start<=x["frame"]<e["frame"] and x["task"]!=e["task"]]
  if other:start=max(start,max(other))
  if stop-start<16:continue
  selected.append((start,stop,e));end=stop
 return selected
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--coverage",type=Path,required=True);ap.add_argument("--features",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 cov=json.loads(a.coverage.read_text());m=json.loads(a.features.read_text());a.output.mkdir(parents=True,exist_ok=False)
 tasks=list(cov["tasks"]);parents={r["episode_id"]:r for r in m["episodes"]};records=[];audit=[];background=[];count=Counter()
 for c in cov["records"]:
  raw=Path(c["actions_path"]);assert sha256(raw)==c["actions_sha256"]
  rows=[json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines()]
  accepted=[];review=[]
  for e in c["events"]:
   e=dict(e);i=e["frame"];item=e["key"].split(":")[-1].removeprefix("minecraft.");base=inventory_count(rows[i-1],item)
   gains=[inventory_count(row,item) for row in rows[i:min(i+21,len(rows))]]
   support=base is not None and any(v is not None and v>base for v in gains)
   e["inventory_gain_within_20_frames"]=support
   e["accepted"]=e["kind"]=="existing_counter_increment" or support
   # Existing increments are observed events; new sparse keys need independent corroboration.
   review.append(e)
   if e["accepted"]:accepted.append(e)
  windows=ranges(accepted,len(rows))
  parent=parents[c["episode_id"]]
  assert parent["source_actions_sha256"]==c["actions_sha256"] and parent["source_video_sha256"]==c["video_sha256"]
  assert parent["source_episode_id"]==c["source_episode_id"] and parent["split"]==c["split"]
  path=a.features.parent/parent["path"];assert sha256(path)==parent["sha256"]
  with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
  assert set(arrays)=={"features","actions","tasks","frames"} and len(arrays["frames"])==len(rows)
  assert np.array_equal(arrays["frames"],np.arange(len(rows)))
  end=0
  for start,stop,e in windows:
   if start>end:background.append(dict(episode_id=c["episode_id"],split=c["split"],start=end,stop=start,task=None,event=None,completion=None,reason="goal unknown; not a failed-task label",parent_path=str(path.resolve()),parent_sha256=parent["sha256"]))
   end=stop
   name=f"clip_{len(records):05d}.npz";p=a.output/name
   np.savez(p,features=arrays["features"][start:stop],actions=arrays["actions"][start:stop],
            frames=arrays["frames"][start:stop],tasks=np.full(stop-start,tasks.index(e["task"]),dtype=np.int64))
   record={k:parent[k] for k in ("source_episode_id","split","origin","source_video_sha256","source_actions_sha256","pairing_protocol")}
   record.update(episode_id=f'{c["episode_id"]}__{start}_{stop}_{e["task"]}',path=name,sha256=sha256(p),
                 parent_episode_id=c["episode_id"],parent_path=str(path.resolve()),parent_sha256=parent["sha256"],start=start,stop=stop,event_evidence=e)
   records.append(record);count[(c["split"],e["task"],"clips")]+=1;count[(c["split"],e["task"],"frames")]+=stop-start
  if end<len(rows):background.append(dict(episode_id=c["episode_id"],split=c["split"],start=end,stop=len(rows),task=None,event=None,completion=None,reason="goal unknown; not a failed-task label",parent_path=str(path.resolve()),parent_sha256=parent["sha256"]))
  audit.append(dict(episode_id=c["episode_id"],events=review))
 result={k:m[k] for k in ("schema_version","origin","action_spec","feature_spec")}
 result.update(tasks=len(tasks),task_names=tasks,policy_mode="task_conditioned",episodes=records,parent_manifest_sha256=sha256(a.features),
               coverage_sha256=sha256(a.coverage),builder_sha256=sha256(__file__),
               task_label_protocol="event_retrospective_128_v1",
               limitations=["Retrospective short-goal labels, not full skill duration or intent ground truth.","Inventory only corroborates offline event; never enters features.","Reward/completion labels not produced: event observation timing needs separate reward protocol."])
 for split in ("train","calibration"):
  for task in tasks:
   if not count[(split,task,"clips")]:raise ValueError(f"No clips: {split}/{task}")
 (a.output/"manifest.json").write_text(json.dumps(result,indent=2))
 (a.output/"event_review.json").write_text(json.dumps(audit,indent=2))
 (a.output/"background.json").write_text(json.dumps(background,indent=2))
 summary={split:{task:{key:count[(split,task,key)] for key in ("clips","frames")} for task in tasks} for split in ("train","calibration")}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
