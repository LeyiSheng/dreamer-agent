"""Inventory-wide training source audit, preserving previous held-out sessions."""
import argparse,json,subprocess,hashlib,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .provenance import sha256
from .sources import session_split
from .event_coverage import scan, DREAMER4_TASKS
from .prepare_task_bc import inventory_count,ranges
from .prepare_reward_labels import clip_labels
def _accept_event(e, rows):
 i=e["frame"];key=e["key"];item=key.split(":")[-1].removeprefix("minecraft.")
 if e["kind"]=="existing_counter_increment":
  return True
 # GUI open / place / use counters often lack a clean inventory-increase corroboration.
 if key.startswith("minecraft.custom:minecraft.interact_with_") or key.startswith("minecraft.use_item:"):
  return True
 base=inventory_count(rows[i-1],item)
 return base is not None and any((v:=inventory_count(row,item)) is not None and v>base for row in rows[i:min(i+21,len(rows))])
def audit(r):
 result=dict(r)
 if r["split"]=="test":return dict(result,status="held_out_not_read")
 try:
  p=Path(r["actions_path"]);raw=p.read_bytes();rows=[json.loads(l) for l in raw.decode("utf-8").splitlines()]
  if len(rows)<2:raise ValueError("Insufficient rows")
  for i,x in enumerate(rows):
   if type(x.get("tick")) is not int or (i and x["tick"]!=rows[i-1]["tick"]+1):raise ValueError("Tick gap")
   if type(x.get("hotbar")) is not int or not 0<=x["hotbar"]<=8:raise ValueError("Unsupported missing/invalid hotbar; requires version adapter")
   if not all(k in x.get("mouse",{}) for k in ("dx","dy","buttons")) or "keys" not in x.get("keyboard",{}):raise ValueError("Unsupported action fields")
  probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,avg_frame_rate,nb_frames","-of","json",r["video_path"]],capture_output=True,text=True,timeout=30)
  if probe.returncode:raise ValueError(probe.stderr.strip()[:200])
  stream=json.loads(probe.stdout)["streams"][0]
  from fractions import Fraction
  if (stream["width"],stream["height"])!=(640,360) or Fraction(stream["avg_frame_rate"])!=20:raise ValueError("Unsupported geometry/fps; requires version adapter")
  if int(stream.get("nb_frames",0)) not in (len(rows),len(rows)+1):raise ValueError("Video/action count mismatch")
  ev=scan(rows)["events"];accepted=[]
  for e in ev:
   e["accepted"]=_accept_event(e, rows)
   if e["accepted"]:accepted.append(e)
  clips=[dict(start=s,stop=e,task=event["task"],background=False) for s,e,event in ranges(accepted,len(rows))]
  # Keep all otherwise-unused frames as event supervision, without inventing goal/completion.
  tasks=list(DREAMER4_TASKS);tail=0;bg=[]
  for clip in clips+[dict(start=len(rows),stop=len(rows))]:
   for start in range(tail,clip["start"],128):
    stop=min(start+128,clip["start"])
    if stop-start>=2:
     tid=int(hashlib.sha256(f'{r["stem"]}:{start}'.encode()).hexdigest()[:8],16)%len(tasks)
     bg.append(dict(start=start,stop=stop,task=tasks[tid],background=True))
   tail=clip["stop"]
  coverage={t:{h:[0,0] for h in ("event","completion")} for t in tasks}
  for clip in clips+bg:
   labels=clip_labels(rows,ev,clip["start"],clip["stop"],clip["task"],tasks.index(clip["task"]))
   for i,l in enumerate(labels[:-1]):
    if i%32==31:continue
    for h in ("event","completion"):
     value=None if h=="completion" and clip["background"] else l[h]
     if value is not None:coverage[clip["task"]][h][value]+=1
  result.update(coverage=coverage,status="eligible_pending_full_decode",frames=len(rows),actions_sha256=hashlib.sha256(raw).hexdigest(),events=ev,clips=sorted(clips+bg,key=lambda c:c["start"]))
 except Exception as e:result.update(status="needs_review",reason=f"{type(e).__name__}: {e}")
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 inv=json.loads(Path("outputs/vpt_inventory_correction_20260915/inventory.json").read_text())
 old=json.loads(Path("outputs/p0_sources_20260913/registry.json").read_text())
 # Conservatively group identical session hashes across aliases/versions.
 priority={"train":0,"calibration":1,"test":2};fixed={}
 for r in old["records"]:
  key=r["source_episode_id"].rsplit("-",1)[-1];sp=r["split"]
  if key not in fixed or priority[sp]>priority[fixed[key]]:fixed[key]=sp
 sources=[]
 for r in inv["records"]:
  rr=dict(r);key=rr["source_session"].rsplit("-",1)[-1];rr["split"]=fixed.get(key,session_split(key,20260913));sources.append(rr)
 (a.output/"discovered.json").write_text(json.dumps(sources))
 counts={}
 with (a.output/"records.jsonl").open("x") as f,ThreadPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(audit,sources)):
   f.write(json.dumps(r)+"\n");f.flush();counts[r["status"]]=counts.get(r["status"],0)+1
   if i%100==0:print(i,counts,flush=True)
 seen={}
 with (a.output/"records.jsonl").open() as src:
  for line in src:
   r=json.loads(line)
   if r["status"]=="eligible_pending_full_decode":seen.setdefault(r["actions_sha256"],set()).add(r["split"])
 collisions={h for h,sp in seen.items() if len(sp)>1}
 with (a.output/"training_records.jsonl").open("x") as dest:
  with (a.output/"records.jsonl").open() as src:
   for line in src:
    r=json.loads(line)
    if r.get("actions_sha256") in collisions:r.update(status="needs_review",reason="Action content crosses splits")
    dest.write(json.dumps(r)+"\n")
 (a.output/"summary.json").write_text(json.dumps(dict(
  counts=counts,total=len(sources),status="audit_complete",
  taskset="dreamer4_table4",tasks=list(DREAMER4_TASKS),
  scope="All local versions; Dreamer 4 Table 4 task events from stats deltas; failures retained for review, no silent deletion; full video decode occurs before each training-source update."),indent=2))
 print("AUDIT_COMPLETE",counts,flush=True)
if __name__=="__main__":main()
