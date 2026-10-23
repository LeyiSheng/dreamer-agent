"""P1 export from reviewed VPT clips. No automatic skill/event labeling.

An annotation JSON declares tasks and clips; each clip has source paths/hashes,
session/split, [start, stop), and a labels JSON with one task per frame and
optional binary event/completion (null means unknown). Only reviewed offline
annotations using vpt_same_index_v1 are accepted.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from .actions import ActionSpec
from .contracts import Episode, validate_dataset
from .provenance import sha256, write_report
from .sources import session_id, require_export_ready
from .vpt_audit import inspect

PROTOCOL = "vpt_same_index_v1"
PREPROCESS = dict(version=1, source_shape=[360,640,3], color="RGB",
                  normalization="float32 /127.5 -1", pad_vertical=[12,12],
                  pad_value=0.0, temporal="independent frame; history in BC GRU",
                  readout="flatten all encoder tokens; no pooling", cursor="raw recorded pixels")


def validate_annotations(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    mode=m.get("policy_mode","task_conditioned")
    if mode not in {"unconditioned","task_conditioned"}:
        raise ValueError("Unknown policy mode")
    basic=mode=="unconditioned"
    if m.get("schema_version") != 1 or m.get("origin") != "offline":
        raise ValueError("Versioned offline data required")
    if (not basic and m.get("reviewed") is not True) or (basic and m.get("source_reviewed") is not True):
        raise ValueError("Reviewed offline annotations required" if not basic else "Reviewed raw sources required")
    if m.get("pairing_protocol") != PROTOCOL or not m.get("reviewer"):
        raise ValueError("Explicit reviewer and VPT pairing protocol required")
    if basic:
        if m.get("tasks") != []:
            raise ValueError("Unconditioned mode has no skill vocabulary")
    elif not isinstance(m.get("tasks"), list) or not m["tasks"] or any(not isinstance(t,str) or not t for t in m["tasks"]) or len(set(m["tasks"])) != len(m["tasks"]):
        raise ValueError("Unique task names required")
    records = m["clips"]
    if not records:
        raise ValueError("No clips")
    validate_dataset([Episode(r["episode_id"],r["source_episode_id"],r["split"],r["origin"]) for r in records])
    ranges, source_splits = {}, {}
    for r in records:
        for name in (("video", "actions") if basic else ("video", "actions", "labels")):
            p = (path.parent/r[name+"_path"]).resolve()
            if sha256(p) != r[name+"_sha256"]:
                raise ValueError("Annotation source hash mismatch: "+name)
            r[name+"_path"] = str(p)
        stem = Path(r["actions_path"]).stem
        if Path(r["video_path"]).stem != stem or session_id(stem) != r["source_episode_id"]:
            raise ValueError("Source pair/session identity mismatch")
        for name in ("video", "actions"):
            fingerprint = (name,r[name+"_sha256"])
            if source_splits.setdefault(fingerprint,r["split"]) != r["split"]:
                raise ValueError("Identical raw source crosses splits")
        start, stop = r["start"], r["stop"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
            raise ValueError("Invalid clip range")
        previous = ranges.setdefault(r["video_sha256"], [])
        if any(start < b and stop > a for a,b in previous):
            raise ValueError("Overlapping annotated clips")
        previous.append((start,stop))
        if basic:
            if "labels_path" in r:
                raise ValueError("Basic BC must not consume skill or reward annotations")
            r["_labels"] = [dict(task=0) for _ in range(stop-start)]
            continue
        labels = json.loads(Path(r["labels_path"]).read_text())
        if len(labels) != stop-start:
            raise ValueError("One annotation per selected frame required")
        for label in labels:
            if set(label) != {"task","event","completion"} or type(label["task"]) is not int or not 0 <= label["task"] < len(m["tasks"]):
                raise ValueError("Invalid task annotation")
            if any(label[k] is not None and (type(label[k]) is not int or label[k] not in (0,1)) for k in ("event","completion")):
                raise ValueError("Binary or null supervision required")
        r["_labels"] = labels
    return m


def decoded_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Video cannot be opened")
    if abs(cap.get(cv2.CAP_PROP_FPS)-20) > .01:
        cap.release()
        raise ValueError("Expected 20 Hz")
    advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape != (360,640,3):
                raise ValueError("Unexpected video geometry")
            count += 1
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if advertised <= 0 or count != advertised:
            raise ValueError("Full decode disagrees with video frame count")
    finally:
        cap.release()


def export(annotations, output, spec, encode_frame, encode_action, extractor_sha256):
    """Callbacks isolate audited backend dependencies for software tests."""
    m = validate_annotations(annotations)
    basic = m.get("policy_mode") == "unconditioned"
    if spec.frame_hz != 20 or spec.action_repeat != 1:
        raise ValueError("Exporter preserves 20Hz timebase")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    records, label_records, dimension = [], [], None
    for index,r in enumerate(m["clips"]):
        audit = inspect(r["actions_path"])
        if not audit["eligible_for_alignment_review"]:
            raise ValueError("Invalid raw action sequence")
        rows = [json.loads(line) for line in Path(r["actions_path"]).read_text().splitlines()]
        if r["stop"] > len(rows):
            raise ValueError("Clip exceeds action labels")
        features, actions, count = [], [], 0
        for t,frame in enumerate(decoded_frames(r["video_path"])):
            count += 1
            if r["start"] <= t < r["stop"]:
                feature = np.asarray(encode_frame(frame), dtype=np.float32)
                if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
                    raise ValueError("Invalid frozen visual output")
                if dimension is None:
                    dimension = len(feature)
                if len(feature) != dimension:
                    raise ValueError("Feature dimension changed")
                features.append(feature)
                actions.append(spec.to_local(np.asarray(encode_action(rows[t]),dtype=np.int64)))
        if count not in (len(rows),len(rows)+1):
            raise ValueError("Only T or T+1 video frames accepted under BC protocol")
        # Detect source mutation during decoding/export.
        for kind in (("video","actions") if basic else ("video","actions","labels")):
            if sha256(r[kind+"_path"]) != r[kind+"_sha256"]:
                raise ValueError("Source changed during export")
        st=Path(r["video_path"]).stat()
        approval=dict(status="approved",export_ready=True,pairing_protocol=PROTOCOL,
            pairing_protocol_verified=True,alignment_verified=False,
            protocol_evidence=str(Path(annotations).resolve()),
            protocol_evidence_sha256=sha256(annotations),video_path=r["video_path"],
            actions_path=r["actions_path"],video_bytes=st.st_size,video_mtime_ns=st.st_mtime_ns,
            video_sha256=r["video_sha256"],actions_sha256=r["actions_sha256"])
        require_export_ready(approval)
        write_report(out/f"source_{index:05d}.json",approval)
        labels = r["_labels"]
        p = out/f"clip_{index:05d}.npz"
        np.savez(p,features=np.stack(features),actions=np.stack(actions),
                 tasks=np.array([l["task"] for l in labels],dtype=np.int64),
                 frames=np.arange(r["start"],r["stop"],dtype=np.int64))
        record = {k:r[k] for k in ("episode_id","source_episode_id","split","origin")}
        record.update(path=p.name,sha256=sha256(p),source_video_sha256=r["video_sha256"],
                      source_actions_sha256=r["actions_sha256"],pairing_protocol=PROTOCOL,
                      decoded_frames=count,unused_tail_frames=count-len(rows))
        records.append(record)
        if not basic:
            lp = out/f"labels_{index:05d}.json"
            write_report(lp, labels)
            label_records.append(dict(episode_id=r["episode_id"],path=lp.name,sha256=sha256(lp)))
        print("exported",r["episode_id"],len(features),flush=True)
    feature_spec = dict(kind="frozen_visual", frozen=True, dimension=dimension,
                        extractor_sha256=extractor_sha256,
                        preprocessing_sha256=__import__("hashlib").sha256(json.dumps(PREPROCESS,sort_keys=True).encode()).hexdigest(),
                        preprocessing=PREPROCESS)
    manifest = dict(schema_version=1,origin="offline",tasks=1 if basic else len(m["tasks"]),task_names=m["tasks"],
                    policy_mode="unconditioned" if basic else "task_conditioned",
                    action_spec=spec.to_dict(),feature_spec=feature_spec,episodes=records,
                    annotations_sha256=sha256(annotations),export_code_sha256=sha256(__file__))
    write_report(out/"manifest.json",manifest)
    if not basic:
        write_report(out/"reward_labels.json",dict(schema_version=1,origin="offline",reviewed=True,
                     feature_manifest_sha256=sha256(out/"manifest.json"),
                     annotations_sha256=sha256(annotations),episodes=label_records))
    return out/"manifest.json"


def load_backend(assets_path, action_probe, device):
    import torch
    assets=json.loads(assets_path.read_text())
    weights=[x for x in assets["files"] if "/tokenizer/" in x["path"]]
    if len(weights)!=2 or any(sha256(x["path"])!=x["sha256"] for x in weights):
        raise ValueError("Verified tokenizer assets required")
    snapshot=Path(weights[0]["path"]).parents[1]
    spec=ActionSpec.from_dict(json.loads(action_probe.read_text())["action_spec"])
    codec_path=Path("Dreamerv4-MC/src/modules/actokenizer.py").resolve()
    if sha256(codec_path)!=spec.codec_sha256:
        raise ValueError("Codec changed")
    backend_spec=importlib.util.spec_from_file_location("_export_codec",codec_path)
    module=importlib.util.module_from_spec(backend_spec)
    backend_spec.loader.exec_module(module)
    codec=module.MineCraftActionTokenizer()
    sys.path.insert(0,str(Path("Dreamerv4-MC").resolve()))
    from src.modules.tokenizer import CausalTokenizer
    tokenizer,info=CausalTokenizer.from_pretrained(str(snapshot),subfolder="tokenizer",
        local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise ValueError("Tokenizer state mismatch")
    tokenizer=tokenizer.to(device).eval().requires_grad_(False)
    @torch.inference_mode()
    def encode(frame):
        x=torch.from_numpy(frame.copy()).permute(2,0,1).float().to(device)/127.5-1
        x=torch.nn.functional.pad(x,(0,0,12,12))[None,None]
        # Full single-frame encoder path: no cached future or previous episode.
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=str(device).startswith("cuda")):
            z=tokenizer.encoder(x)
        return z.float().cpu().numpy().reshape(-1)
    def action(row):
        if type(row.get("hotbar")) is not int or not 0 <= row["hotbar"] <= 8:
            raise ValueError("Explicit valid hotbar action state required")
        return codec.get_action_index_from_actiondict(codec.json_action_to_env_action(row,hotbar=True)[0],include_gui=True)
    import hashlib
    extractor=hashlib.sha256(json.dumps(dict(assets=weights,backend=sha256("Dreamerv4-MC/src/modules/tokenizer.py"),
             exporter=sha256(__file__)),sort_keys=True).encode()).hexdigest()
    return spec,encode,action,extractor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--assets",type=Path,default=Path("outputs/acceptance_20260913/assets.json"))
    p.add_argument("--action-probe",type=Path,default=Path("outputs/acceptance_20260913/action_probe.json"))
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    validate_annotations(a.annotations)
    spec,encode,action,extractor=load_backend(a.assets,a.action_probe,a.device)
    export(a.annotations,a.output,spec,encode,action,extractor)


if __name__=="__main__":
    main()
