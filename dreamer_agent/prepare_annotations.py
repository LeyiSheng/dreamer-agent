"""Prepare an UNREVIEWED annotation template from screened offline sources."""
import argparse
import json
from pathlib import Path
from .provenance import sha256,write_report
from .vpt_audit import inspect
from .export_vpt import PROTOCOL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=64)
    p.add_argument("--per-split",type=int,default=1)
    a=p.parse_args()
    if min(a.frames,a.per_split)<1:
        raise ValueError("Positive sample sizes required")
    a.output.mkdir(parents=True,exist_ok=False)
    registry=json.loads(a.registry.read_text())
    clips=[];seen=set()
    for split in ("train","calibration"):
        count=0
        for r in sorted(registry["records"],key=lambda r:r["episode_id"]):
            if r["split"]!=split or r["status"]=="rejected" or r["source_episode_id"] in seen:
                continue
            audit=inspect(r["actions_path"])
            if not audit["eligible_for_alignment_review"]:
                continue
            if audit["sha256"]!=r["actions_sha256"]:
                raise ValueError("Registry source changed")
            stop=min(a.frames,audit["valid_rows"])
            lp=a.output/f"labels_{len(clips):03d}.json"
            write_report(lp,[dict(task=None,event=None,completion=None) for _ in range(stop)])
            clip={k:r[k] for k in ("episode_id","source_episode_id","split","origin")}
            for name in ("video","actions"):
                clip[name+"_path"]=r[name+"_path"]
                clip[name+"_sha256"]=sha256(r[name+"_path"])
            clip.update(labels_path=lp.name,labels_sha256=sha256(lp),start=0,stop=stop)
            clips.append(clip);seen.add(r["source_episode_id"]);count+=1
            if count>=a.per_split:
                break
    write_report(a.output/"annotations.json",dict(schema_version=1,origin="offline",
        reviewed=False,reviewer="",pairing_protocol=PROTOCOL,tasks=[],clips=clips,
        scope="Unreviewed template only. Supply task names and labels, review raw video, refresh label hashes before enabling export."))
    print("Unreviewed clips:",len(clips))


if __name__=="__main__":
    main()
