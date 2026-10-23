"""Raw source screening; readability never implies training approval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from .provenance import sha256, write_report


def session_id(stem):
    match = re.fullmatch(r"(.+)-([a-f0-9]{12})-(\d{8})-(\d{6})", stem)
    if not match:
        raise ValueError("Unrecognized VPT filename")
    return match[1]+"-"+match[2]


def session_split(session, seed):
    bucket = int(hashlib.sha256(f"{seed}:{session}".encode()).hexdigest()[:16],16)%100
    return "train" if bucket<80 else "calibration" if bucket<90 else "test"


def screen(actions, videos, seed=20260913):
    actions=Path(actions)
    video=Path(videos)/(actions.stem+".mp4")
    record=dict(episode_id=actions.stem,actions_path=str(actions.resolve()),video_path=str(video.resolve()),
                origin="offline",export_ready=False,alignment_verified=False,status="rejected")
    try:
        record["source_episode_id"]=session_id(actions.stem)
        record["split"]=session_split(record["source_episode_id"],seed)
        record["actions_sha256"]=sha256(actions)
        st=video.stat()
        record.update(video_bytes=st.st_size,video_mtime_ns=st.st_mtime_ns)
        p=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
             "stream=width,height,avg_frame_rate,nb_frames","-of","json",str(video)],
             capture_output=True,text=True,timeout=30)
        if p.returncode:
            raise ValueError(p.stderr.strip())
        streams=json.loads(p.stdout)["streams"]
        if len(streams)!=1 or Fraction(streams[0]["avg_frame_rate"])!=20:
            raise ValueError("Unexpected stream count or frame rate")
        if int(streams[0].get("nb_frames",0))<=0:
            raise ValueError("Frame count unavailable")
        record.update(status="alignment_pending",stream=streams[0])
    except (ValueError,ZeroDivisionError,KeyError,OSError,subprocess.TimeoutExpired) as exc:
        record["reason"]=str(exc)
    return record


def require_export_ready(record):
    if record.get("status")!="approved" or record.get("export_ready") is not True:
        raise ValueError("Source is not approved for training export")
    # Protocol review is distinct from a measured physical lag.
    if record.get("pairing_protocol") == "vpt_same_index_v1" and record.get("pairing_protocol_verified") is True:
        evidence=Path(record["protocol_evidence"])
        expected=record["protocol_evidence_sha256"]
    elif record.get("alignment_verified") is True:
        evidence=Path(record["alignment_evidence"])
        expected=record["alignment_evidence_sha256"]
    else:
        raise ValueError("Reviewed pairing protocol or alignment evidence required")
    if sha256(evidence)!=expected:
        raise ValueError("Pairing evidence changed")
    video=Path(record["video_path"])
    st=video.stat()
    if st.st_size!=record["video_bytes"] or st.st_mtime_ns!=record["video_mtime_ns"]:
        raise ValueError("Source changed after review")
    if sha256(video)!=record["video_sha256"] or sha256(record["actions_path"])!=record["actions_sha256"]:
        raise ValueError("Source content hash mismatch")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actions",type=Path,required=True)
    p.add_argument("--videos",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20260913)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    files=sorted(a.actions.glob("*.jsonl"))
    if not files:raise ValueError("No sources")
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(lambda f:screen(f,a.videos,a.seed),files))
    report=dict(schema_version=1,seed=a.seed,records=records,
        rejected=sum(r["status"]=="rejected" for r in records),
        sessions=len({r["source_episode_id"] for r in records if "source_episode_id" in r}),
        scope="Container screening and proposed session splits; no source approved for export")
    write_report(a.output,report)
    print(f"sources={len(records)} rejected={report['rejected']} sessions={report['sessions']}")


if __name__=="__main__":
    main()
