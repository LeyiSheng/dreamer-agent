"""Roll official VPT bc-early-game-3x inside the frozen Dreamer WM for a short clip.

No MineRL/Java Minecraft: observation comes from the world model after a real
VPT context prefill. This is an imagination diagnostic, not real-env evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VPT_ROOT = ROOT / "third_party" / "Video-Pre-Training"
DEPS = ROOT / "third_party" / "vpt_deps"
for path in (str(DEPS), str(VPT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from gym3.types import DictType  # noqa: E402
from agent import ACTION_TRANSFORMER_KWARGS, AGENT_RESOLUTION  # noqa: E402
from lib.action_mapping import CameraHierarchicalMapping  # noqa: E402
from lib.actions import ActionTransformer  # noqa: E402
from lib.policy import MinecraftAgentPolicy  # noqa: E402
from lib.torch_util import set_default_torch_device  # noqa: E402

from .actions import ActionSpec
from .provenance import sha256, write_report
from .world_model import FrozenWorldAdapter


def _resize_agent(rgb):
    return cv2.resize(rgb, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)


def _env_to_tokens(codec, env_action):
    action = {k: (int(v) if k != "camera" else np.asarray(v, dtype=np.float32))
              for k, v in env_action.items()}
    if "camera" not in action:
        action["camera"] = np.zeros(2, dtype=np.float32)
    return codec.get_action_index_from_actiondict(action, include_gui=True)


def _load_vpt(model_path, weights_path, device):
    params = pickle.load(open(model_path, "rb"))
    net = params["model"]["args"]["net"]["args"]
    pi_head = dict(params["model"]["args"]["pi_head_opts"])
    pi_head["temperature"] = float(pi_head["temperature"])
    set_default_torch_device(torch.device(device))
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    space = DictType(**mapper.get_action_space_update())
    policy = MinecraftAgentPolicy(
        policy_kwargs=net, pi_head_kwargs=pi_head, action_space=space).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Weight mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    policy.eval().requires_grad_(False)
    return policy, mapper, ActionTransformer(**ACTION_TRANSFORMER_KWARGS)


def record(
    *,
    weights: Path,
    model: Path,
    output: Path,
    video: Path,
    actions: Path,
    context: int = 32,
    horizon: int = 400,
    steps: int = 4,
    seed: int = 0,
    start_frame: int = 0,
    device: str = "cuda:6",
    assets: Path = Path("outputs/acceptance_20260913/assets.json"),
    action_probe: Path = Path("outputs/acceptance_20260913/action_probe.json"),
):
    if context < 1 or horizon < 1 or start_frame < 0:
        raise ValueError("Invalid lengths")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy, mapper, action_xf = _load_vpt(model, weights, device)
    hidden = policy.initial_state(1)
    dummy_first = torch.tensor([False], device=device)

    assets_data = json.loads(Path(assets).read_text())
    snapshot = Path(assets_data["files"][0]["path"]).parents[1]
    sys.path.insert(0, str((ROOT / "Dreamerv4-MC").resolve()))
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
    if sha256(ROOT / "Dreamerv4-MC/src/modules/actokenizer.py") != spec.codec_sha256:
        raise ValueError("Codec changed")
    adapter = FrozenWorldAdapter(tokenizer, dynamics, generate_one_frame, spec, steps=steps, seed=seed)
    codec = MineCraftActionTokenizer()

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
    with Path(actions).open() as stream:
        for _ in range(start_frame):
            next(stream)
        rows = [json.loads(next(stream)) for _ in range(context)]

    ctx_tokens = torch.tensor([
        codec.get_action_index_from_actiondict(
            codec.json_action_to_env_action(row, hotbar=True)[0], include_gui=True)
        for row in rows], device=device, dtype=torch.long)
    pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
    pixels = torch.nn.functional.pad(pixels, (0, 0, 12, 12))[None].to(device)
    adapter.reset(pixels, ctx_tokens, seed=seed)

    # Warm VPT recurrent state on real context (obs only).
    with torch.inference_mode():
        for frame in frames:
            obs = {"img": torch.from_numpy(_resize_agent(frame)[None]).to(device)}
            _, hidden, _ = policy.act(obs, dummy_first, hidden, stochastic=False)

    out_frames = [f.copy() for f in frames]
    env_actions = []
    current = frames[-1]
    with torch.inference_mode():
        for step in range(horizon):
            obs = {"img": torch.from_numpy(_resize_agent(current)[None]).to(device)}
            agent_act, hidden, _ = policy.act(obs, dummy_first, hidden, stochastic=True)
            agent_np = {
                "buttons": agent_act["buttons"].detach().cpu().numpy(),
                "camera": agent_act["camera"].detach().cpu().numpy(),
            }
            env_act = action_xf.policy2env(mapper.to_factored(agent_np))
            flat = {}
            for k, v in env_act.items():
                arr = np.asarray(v)
                if k == "camera":
                    flat[k] = arr.reshape(-1)[:2].astype(np.float32)
                else:
                    flat[k] = int(np.asarray(v).reshape(-1)[0])
            env_actions.append({
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in flat.items()})
            tokens = torch.as_tensor(_env_to_tokens(codec, flat), device=device, dtype=torch.long).unsqueeze(0)
            _, frame = adapter.step(tokens)
            rgb = ((frame[0, 0].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            if rgb.shape[0] == 384:
                rgb = rgb[12:-12]
            current = rgb.copy()
            out_frames.append(current)
            if (step + 1) % 20 == 0:
                print(f"STEP {step+1}/{horizon}", flush=True)

    mp4 = output / "trajectory.mp4"
    h, w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    try:
        for frame in out_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    # Prefer h264 for playback if ffmpeg is available.
    h264 = output / "trajectory_h264.mp4"
    rc = os.system(
        f'ffmpeg -y -loglevel error -i "{mp4}" -c:v libx264 -pix_fmt yuv420p "{h264}"')
    actions_path = output / "vpt_actions.jsonl"
    with actions_path.open("w") as f:
        for row in env_actions:
            f.write(json.dumps(row) + "\n")
    report = dict(
        schema_version=1,
        kind="vpt_early_game_imagination_trajectory",
        weights=str(Path(weights).resolve()),
        weights_sha256=sha256(weights),
        model=str(Path(model).resolve()),
        model_sha256=sha256(model),
        video=str((h264 if rc == 0 and h264.exists() else mp4).resolve()),
        video_sha256=sha256(h264 if rc == 0 and h264.exists() else mp4),
        actions_path=str(actions_path.resolve()),
        context_frames=context,
        imagined_frames=horizon,
        total_frames=len(out_frames),
        seconds_approx=round((context + horizon) / 20, 2),
        source_video=str(Path(video).resolve()),
        source_actions=str(Path(actions).resolve()),
        start_frame=start_frame,
        seed=seed,
        steps=steps,
        device=device,
        note="Closed-loop inside frozen Dreamer WM; not real Minecraft evaluation. No MineRL.",
        ffmpeg_h264=rc == 0,
    )
    write_report(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path,
                   default=Path("checkpoints/bc-early-game-3x.weights"))
    p.add_argument("--model", type=Path,
                   default=Path("checkpoints/foundation-model-3x.model"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video", type=Path,
                   default=Path("data/vpt/raw/videos/cheeky-cornflower-setter-02e496ce4abb-20220421-093149.mp4"))
    p.add_argument("--actions", type=Path,
                   default=Path("data/vpt/raw/actions/cheeky-cornflower-setter-02e496ce4abb-20220421-093149.jsonl"))
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--horizon", type=int, default=400, help="Imagined frames at 20 FPS (~20s default)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--device", default="cuda:6")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    record(**{k.replace("-", "_"): v for k, v in vars(a).items()})


if __name__ == "__main__":
    main()
