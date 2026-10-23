"""Prepare task-free P1a source manifest from audited VPT recordings."""
import argparse
import hashlib
import json
from pathlib import Path
from .provenance import sha256, write_report
from .sources import session_id
from .vpt_audit import inspect
from .export_vpt import PROTOCOL, validate_annotations


def prepare(registry, output, *, train_sessions=4, calibration_sessions=2, frames=256, seed=20260914):
    if min(train_sessions,calibration_sessions,frames)<1:
        raise ValueError("Positive sample sizes required")
    registry=Path(registry)
    data=json.loads(registry.read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    chosen=[];seen=set();audits=[]
    for split,limit in (("train",train_sessions),("calibration",calibration_sessions)):
        # Sample by identity only, never rewards, labels or image/model scores.
        rows=sorted((r for r in data["records"] if r.get("split")==split and r.get("status")!="rejected"),
            key=lambda r:hashlib.sha256(f"{seed}:{r['episode_id']}".encode()).hexdigest())
        count=0
        for r in rows:
            if r["source_episode_id"] in seen:
                continue
            video,actions=Path(r["video_path"]),Path(r["actions_path"])
            if session_id(actions.stem)!=r["source_episode_id"] or video.stem!=actions.stem:
                raise ValueError("Source identity changed")
            st=video.stat()
            if st.st_size!=r["video_bytes"] or st.st_mtime_ns!=r["video_mtime_ns"]:
                raise ValueError("Video changed since screening")
            audit=inspect(actions)
            if audit["sha256"]!=r["actions_sha256"]:
                raise ValueError("Actions changed since screening")
            if not audit["eligible_for_alignment_review"] or audit["valid_rows"]<frames:
                audits.append(dict(episode=r["episode_id"],eligible=False,audit=audit))
                continue
            start=int(hashlib.sha256(f"{seed}:start:{r['episode_id']}".encode()).hexdigest()[:16],16)%(audit["valid_rows"]-frames+1)
            clip={k:r[k] for k in ("episode_id","source_episode_id","split","origin")}
            clip.update(start=start,stop=start+frames,video_path=str(video.resolve()),actions_path=str(actions.resolve()),
                        video_sha256=sha256(video),actions_sha256=audit["sha256"])
            chosen.append(clip);seen.add(r["source_episode_id"]);count+=1
            audits.append(dict(episode=r["episode_id"],eligible=True,audit=audit))
            print("selected",split,r["episode_id"],start,frames,flush=True)
            if count==limit:
                break
        if count!=limit:
            raise ValueError("Insufficient independent valid sessions for "+split)
    manifest=dict(schema_version=1,origin="offline",policy_mode="unconditioned",
        source_reviewed=True,reviewer="automated raw identity/hash/tick audit; not skill review",
        pairing_protocol=PROTOCOL,tasks=[],clips=chosen,selection_seed=seed,
        selection_rule="identity hash order, unique original sessions, uniform hash-derived contiguous clip",
        registry_sha256=sha256(registry),source_split_seed=data["seed"],
        code_sha256=sha256(__file__),scope="Small real-data launch set; no skill/event labels; no test samples")
    write_report(out/"raw_audit.json",audits)
    p=out/"sources.json";write_report(p,manifest)
    validate_annotations(p)
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--train-sessions",type=int,default=4)
    p.add_argument("--calibration-sessions",type=int,default=2)
    p.add_argument("--frames",type=int,default=256)
    p.add_argument("--seed",type=int,default=20260914)
    prepare(**vars(p.parse_args()))


if __name__=="__main__":
    main()
