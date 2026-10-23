"""Record one imagined video trajectory from a trained BC checkpoint + frozen WM.

Prefills with real VPT context, then rolls the HistoryPolicy inside FrozenWorldAdapter.
This is an imagination diagnostic clip, not real-environment evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from .actions import ActionSpec
from .learning import HistoryPolicy
from .provenance import sha256, write_report
from .world_model import FrozenWorldAdapter


def _one_hot_past(local, action_sizes):
    past = np.zeros(sum(action_sizes), dtype=np.float32)
    offset = 0
    for k, n in enumerate(action_sizes):
        past[offset + int(local[k])] = 1
        offset += n
    return past


def _encode_feature(tokenizer, frame_rgb, device):
    x = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1).float().to(device) / 127.5 - 1
    x = torch.nn.functional.pad(x, (0, 0, 12, 12))[None, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = tokenizer.encoder(x)
    return z.float().reshape(-1).cpu()


def record(checkpoint, output, *, context=32, horizon=64, steps=4, seed=0,
           assets=Path("outputs/acceptance_20260913/assets.json"),
           action_probe=Path("outputs/acceptance_20260913/action_probe.json"),
           video=None, actions=None, task=0, start_frame=0, device="cuda"):
    if context < 1 or horizon < 1 or start_frame < 0 or task < 0:
        raise ValueError("Invalid record parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]
    config = metadata["config"]
    if task >= int(config["tasks"]):
        raise ValueError("Task id outside checkpoint vocabulary")
    policy = HistoryPolicy(**config)
    policy.load_state_dict(ckpt["model"])
    policy = policy.to(device).eval()
    assets_data = json.loads(Path(assets).read_text())
    snapshot = Path(assets_data["files"][0]["path"]).parents[1]
    import sys
    sys.path.insert(0, str(Path("Dreamerv4-MC").resolve()))
    from src.modules.tokenizer import CausalTokenizer
    from src.modules.dynamic_model import DynamicModel
    from src.modules.actokenizer import MineCraftActionTokenizer
    from src.inference.infer import generate_one_frame
    tokenizer, info_t = CausalTokenizer.from_pretrained(
        str(snapshot), subfolder="tokenizer", local_files_only=True,
        torch_dtype=torch.bfloat16, output_loading_info=True)
    dynamics, info_d = DynamicModel.from_pretrained(
        str(snapshot), subfolder="dynamic", local_files_only=True,
        torch_dtype=torch.bfloat16, output_loading_info=True)
    for name, info in (("tokenizer", info_t), ("dynamic", info_d)):
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("State mismatch: " + name)
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    dynamics = dynamics.to(device).eval().requires_grad_(False)
    spec = ActionSpec.from_dict(json.loads(Path(action_probe).read_text())["action_spec"])
    if sha256(Path("Dreamerv4-MC/src/modules/actokenizer.py")) != spec.codec_sha256:
        raise ValueError("Codec changed")
    if tuple(spec.action_sizes) != tuple(config["action_sizes"]):
        raise ValueError("Action sizes mismatch vs checkpoint")
    adapter = FrozenWorldAdapter(tokenizer, dynamics, generate_one_frame, spec, steps=steps, seed=seed)
    if video is None:
        video = Path("data/vpt/raw/videos/cheeky-cornflower-setter-1d6469cf1ed4-20220421-150148.mp4")
    if actions is None:
        actions = Path("data/vpt/raw/actions") / (Path(video).stem + ".jsonl")
    video, actions = Path(video), Path(actions)
    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(context):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("Insufficient context frames")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    with actions.open() as stream:
        for _ in range(start_frame):
            next(stream)
        rows = [json.loads(next(stream)) for _ in range(context)]
    codec = MineCraftActionTokenizer()
    ctx_tokens = torch.tensor([
        codec.get_action_index_from_actiondict(
            codec.json_action_to_env_action(row, hotbar=True)[0], include_gui=True)
        for row in rows], device=device, dtype=torch.long)
    pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
    pixels = torch.nn.functional.pad(pixels, (0, 0, 12, 12))[None].to(device)
    adapter.reset(pixels, ctx_tokens, seed=seed)
    # Warm policy history on real context (zero past at t=0, then recorded locals).
    features = []
    past_list = []
    locals_hist = []
    for t, frame in enumerate(frames):
        feat = _encode_feature(tokenizer, frame, torch.device(device))
        features.append(feat)
        if t == 0:
            past_list.append(torch.zeros(sum(spec.action_sizes)))
        else:
            past_list.append(torch.from_numpy(_one_hot_past(locals_hist[-1], spec.action_sizes)))
        local = spec.to_local(ctx_tokens[t].detach().cpu().numpy())
        locals_hist.append(local)
    out_frames = [f.copy() for f in frames]
    local_actions = []
    with torch.inference_mode():
        for step in range(horizon):
            feat_t = torch.stack(features[-context:]).unsqueeze(0).to(device)
            past_t = torch.stack(past_list[-context:]).unsqueeze(0).to(device)
            tasks = torch.full((1, feat_t.shape[1]), int(task), dtype=torch.long, device=device)
            logits = policy(feat_t, past_t, tasks)
            # Execute prediction distance zero (current action).
            local = np.array([scores[0, -1, 0].argmax(-1).item() for scores in logits], dtype=np.int64)
            local_actions.append(local.tolist())
            tokens = torch.as_tensor(spec.to_tokens(local), device=device, dtype=torch.long).unsqueeze(0)
            latent, frame = adapter.step(tokens)
            rgb = ((frame[0, 0].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            # Decoder emits padded 384x640; crop vertical pad for display/encode match.
            if rgb.shape[0] == 384:
                rgb = rgb[12:-12]
            out_frames.append(rgb.copy())
            feat = _encode_feature(tokenizer, rgb, torch.device(device))
            features.append(feat)
            past_list.append(torch.from_numpy(_one_hot_past(local, spec.action_sizes)))
            del latent
    mp4 = output / "trajectory.mp4"
    h, w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    try:
        for frame in out_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    report = dict(
        schema_version=1,
        kind="imagination_bc_trajectory",
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint),
        video=str(mp4.resolve()),
        video_sha256=sha256(mp4),
        context_frames=context,
        imagined_frames=horizon,
        total_frames=len(out_frames),
        source_video=str(video.resolve()),
        source_actions=str(actions.resolve()),
        task_id=int(task),
        start_frame=int(start_frame),
        policy_mode=metadata.get("policy_mode"),
        seed=seed,
        steps=steps,
        note="Imagination diagnostic; not real MC evaluation",
        local_actions=local_actions,
    )
    write_report(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "local_actions"}), flush=True)
    return mp4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--assets", type=Path, default=Path("outputs/acceptance_20260913/assets.json"))
    p.add_argument("--action-probe", type=Path, default=Path("outputs/acceptance_20260913/action_probe.json"))
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--actions", type=Path, default=None)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    record(**vars(a))


if __name__ == "__main__":
    main()
