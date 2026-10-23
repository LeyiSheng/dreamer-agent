"""Prepare full-segment unconditioned BC sources from audited VPT recordings.

Uses every eligible train/calibration complete segment (start=0, stop=valid_rows).
Invalid sources are excluded with recorded reasons. Test split is never included.
"""
import argparse
import json
from pathlib import Path
from .provenance import sha256, write_report
from .sources import session_id
from .vpt_audit import inspect
from .export_vpt import PROTOCOL, validate_annotations


def prepare(registry, output):
    registry = Path(registry)
    data = json.loads(registry.read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    chosen, excluded, audits = [], [], []
    for r in data["records"]:
        split = r.get("split")
        if split == "test" or r.get("status") == "rejected":
            continue
        if split not in {"train", "calibration"}:
            excluded.append(dict(episode_id=r["episode_id"], reason="unknown_split", split=split))
            continue
        video, actions = Path(r["video_path"]), Path(r["actions_path"])
        try:
            if session_id(actions.stem) != r["source_episode_id"] or video.stem != actions.stem:
                raise ValueError("Source identity changed")
            if not video.is_file():
                raise ValueError("Missing video")
            st = video.stat()
            if st.st_size != r["video_bytes"] or st.st_mtime_ns != r["video_mtime_ns"]:
                raise ValueError("Video changed since screening")
            audit = inspect(actions)
            if audit["sha256"] != r["actions_sha256"]:
                raise ValueError("Actions changed since screening")
            if not audit["eligible_for_alignment_review"] or audit["valid_rows"] < 1:
                reason = audit.get("reason") or "ineligible_or_empty"
                excluded.append(dict(episode_id=r["episode_id"], reason=reason, split=split, audit=audit))
                audits.append(dict(episode=r["episode_id"], eligible=False, audit=audit))
                continue
            clip = {k: r[k] for k in ("episode_id", "source_episode_id", "split", "origin")}
            clip.update(
                start=0,
                stop=int(audit["valid_rows"]),
                video_path=str(video.resolve()),
                actions_path=str(actions.resolve()),
                video_sha256=sha256(video),
                actions_sha256=audit["sha256"],
            )
            chosen.append(clip)
            audits.append(dict(episode=r["episode_id"], eligible=True, audit=audit))
            print("selected", split, r["episode_id"], 0, audit["valid_rows"], flush=True)
        except Exception as exc:
            excluded.append(dict(
                episode_id=r["episode_id"], split=split,
                reason=f"{type(exc).__name__}: {exc}"))
            print("excluded", r["episode_id"], type(exc).__name__, flush=True)
    if not chosen:
        raise ValueError("No eligible full segments")
    train_n = sum(1 for c in chosen if c["split"] == "train")
    cal_n = sum(1 for c in chosen if c["split"] == "calibration")
    frames = sum(c["stop"] - c["start"] for c in chosen)
    manifest = dict(
        schema_version=1, origin="offline", policy_mode="unconditioned",
        source_reviewed=True,
        reviewer="automated raw identity/hash/tick audit; not skill review",
        pairing_protocol=PROTOCOL, tasks=[], clips=chosen,
        selection_rule="all eligible train/calibration complete segments; start=0 stop=valid_rows",
        registry_sha256=sha256(registry), source_split_seed=data["seed"],
        code_sha256=sha256(__file__),
        scope="Full available-data unconditioned BC; no skill/event labels; no test samples",
        counts=dict(train=train_n, calibration=cal_n, excluded=len(excluded), frames=frames),
    )
    write_report(out / "raw_audit.json", audits)
    write_report(out / "excluded.json", excluded)
    p = out / "sources.json"
    write_report(p, manifest)
    validate_annotations(p)
    print(json.dumps(manifest["counts"]), flush=True)
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
