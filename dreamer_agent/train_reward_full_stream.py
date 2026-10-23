"""Stream all audited local sources through frozen tokenizer/BC and corrected reward head.

Supports --resume from latest.pt and multi-GPU source sharding.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .export_vpt import PREPROCESS, decoded_frames, load_backend
from .learning import HistoryPolicy
from .prepare_reward_labels import clip_labels
from .provenance import sha256
from .reward import TemporalRewardCompletionHeads
from .train_reward import calibration_with_confirmation

TASKS = ["obtain_log", "craft_planks", "craft_crafting_table", "craft_wooden_pickaxe", "obtain_cobblestone"]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--audit", type=Path, required=True)
  ap.add_argument("--output", type=Path, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--resume", type=Path, default=None)
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--shard-id", type=int, default=0)
  ap.add_argument("--device", default="cuda")
  a = ap.parse_args()
  if not (0 <= a.shard_id < a.num_shards):
    raise ValueError("shard_id out of range")

  torch.set_num_threads(1)
  torch.manual_seed(a.seed + a.shard_id * 1009)
  summary = json.loads((a.audit / "summary.json").read_text())
  assert summary["status"] == "audit_complete"
  records = [json.loads(l) for l in (a.audit / "training_records.jsonl").read_text().splitlines()]
  eligible = [r for r in records if r["status"] == "eligible_pending_full_decode"]
  train = [r for r in eligible if r["split"] == "train"]
  cal = [r for r in eligible if r["split"] == "calibration"]
  if not train or not cal:
    raise ValueError("Missing train/calibration sources")

  device = torch.device(a.device if torch.cuda.is_available() else "cpu")
  spec, encode, action, extractor = load_backend(
    Path("outputs/acceptance_20260913/assets.json"),
    Path("outputs/acceptance_20260913/action_probe.json"),
    str(device))
  bp = Path(f"outputs/p1b_task_bc_v1/seed_{a.seed}/bc.pt")
  bc = torch.load(bp, map_location="cpu", weights_only=True)
  if bc["metadata"]["action_spec_sha256"] != spec.fingerprint or bc["metadata"]["feature_spec"]["extractor_sha256"] != extractor:
    raise ValueError("Frozen feature/action version mismatch")
  policy = HistoryPolicy(**bc["metadata"]["config"]).to(device).eval().requires_grad_(False)
  policy.load_state_dict(bc["model"])

  cp = Path(f"outputs/p1c_retrain_small_v2/seed_{a.seed}/fixed_best.pt")
  checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
  if checkpoint["provenance"]["bc_sha256"] != sha256(bp):
    raise ValueError("Small checkpoint BC mismatch")
  head = TemporalRewardCompletionHeads(128, 5, 128).to(device)
  head.load_state_dict(checkpoint["model"])

  start_source = 0
  updates = 0
  if a.resume is not None:
    prior = torch.load(a.resume, map_location="cpu", weights_only=False)
    head.load_state_dict(prior["model"])
    start_source = int(prior.get("next_source_index", 0))
    updates = int(prior.get("updates", 0))
    print(f"RESUME start={start_source} updates={updates} shard={a.shard_id}/{a.num_shards}", flush=True)

  opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
  if a.resume is not None and "optimizer" in prior:
    try:
      opt.load_state_dict(prior["optimizer"])
    except Exception as exc:
      print("OPT_RESUME_SKIP", type(exc).__name__, str(exc)[:120], flush=True)

  weights = {h: [] for h in ("event", "completion")}
  for h in weights:
    for t in TASKS:
      neg = sum(r["coverage"][t][h][0] for r in train)
      pos = sum(r["coverage"][t][h][1] for r in train)
      if not neg or not pos:
        raise ValueError("Unsupported full-corpus task/head")
      weights[h].append(min(10.0, (neg / pos) ** 0.5))
  frequency = np.array([sum(sum(c["task"] == t for c in r["clips"]) for r in train) for t in TASKS])
  taskweight = frequency.sum() / (5 * frequency)

  out = a.output
  out.mkdir(parents=True, exist_ok=(a.resume is not None))
  meta = dict(
    seed=a.seed, epochs=1,
    scope="all audited eligible train sources, one streaming pass; all eligible calibration after training; test untouched",
    audit_sha256=sha256(a.audit / "training_records.jsonl"), bc_sha256=sha256(bp),
    init_sha256=sha256(cp), script_sha256=sha256(__file__), plan_sha256=sha256("docs/PLAN_V6.md"),
    extractor_sha256=extractor, train_sources=len(train), calibration_sources=len(cal),
    weights=weights, task_loss_weights=taskweight.tolist(), rl_ready=False,
    feature_cache="current source only; no corpus-sized cache",
    num_shards=a.num_shards, shard_id=a.shard_id, resume_start_source=start_source,
    resume=str(a.resume.resolve()) if a.resume else None)
  (out / "config.json").write_text(json.dumps(meta, indent=2))

  def prepare(r):
    raw = Path(r["actions_path"])
    assert sha256(raw) == r["actions_sha256"]
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines()]
    locals_ = np.stack([spec.to_local(np.asarray(action(row), dtype=np.int64)) for row in rows])
    vp = Path(r["video_path"])
    vh = sha256(vp)
    features = []
    for i, frame in enumerate(decoded_frames(vp)):
      if i < len(rows):
        features.append(encode(frame))
    if len(features) != len(rows):
      raise ValueError("Incomplete video decode")
    if sha256(raw) != r["actions_sha256"] or sha256(vp) != vh:
      raise ValueError("Source changed during encoding")
    features = np.stack(features)
    items = []
    for clip in r["clips"]:
      start, stop = clip["start"], clip["stop"]
      f = features[start:stop]
      actions = locals_[start:stop]
      l = len(f)
      past = np.zeros((l, sum(spec.action_sizes)), np.float32)
      offset = 0
      for k, n in enumerate(spec.action_sizes):
        past[1:, offset:offset + n] = np.eye(n, dtype=np.float32)[actions[:-1, k]]
        offset += n
      hs = []
      with torch.no_grad():
        for j in range(0, l, 32):
          hs.append(policy.encode_history(
            torch.from_numpy(f[j:j + 32])[None].to(device),
            torch.from_numpy(past[j:j + 32])[None].to(device))[0])
      h = torch.cat(hs)
      tid = TASKS.index(clip["task"])
      ls = clip_labels(rows, r["events"], start, stop, clip["task"], tid)
      y = {}
      m = {}
      for name in ("event", "completion"):
        ys = [None if name == "completion" and clip["background"] else x[name] for x in ls[:-1]]
        y[name] = torch.tensor([[x or 0 for x in ys]], dtype=torch.float32, device=device)
        m[name] = torch.tensor([[x is not None and i % 32 != 31 for i, x in enumerate(ys)]], device=device)
      items.append((h[:-1][None], h[1:][None], torch.full((1, l - 1), tid, dtype=torch.long, device=device), y, m, tid))
    return items, vh

  seen_video_hashes = set()
  order = list(train)
  random.Random(a.seed + 20260915).shuffle(order)
  failures = 0
  shard_seen = 0
  begin = start_source
  for idx in range(begin, len(order)):
    if (idx - begin) % a.num_shards != a.shard_id:
      continue
    r = order[idx]
    try:
      items, vh = prepare(r)
    except (ValueError, OSError) as exc:
      failures += 1
      with (out / "failed_sources.jsonl").open("a") as f:
        f.write(json.dumps(dict(stem=r["stem"], shard_id=a.shard_id, error=f"{type(exc).__name__}: {exc}")) + "\n")
      print("SOURCE_FAILED", r["stem"], str(exc)[:150], flush=True)
      continue
    if vh in seen_video_hashes:
      continue
    seen_video_hashes.add(vh)
    random.Random(a.seed + idx).shuffle(items)
    head.train()
    losses = []
    for h, n, t, y, m, tid in items:
      z = head(h, n, t)
      terms = []
      for k in y:
        if m[k].any():
          v = torch.nn.functional.binary_cross_entropy_with_logits(z[k], y[k], reduction="none")
          v = v * torch.where(y[k] > 0, weights[k][tid], 1.0)
          terms.append(v[m[k]].mean())
      if not terms:
        continue
      loss = torch.stack(terms).mean() * float(taskweight[tid])
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
      opt.step()
      updates += 1
      losses.append(float(loss.detach()))
    shard_seen += 1
    row = dict(source_index=idx, stem=r["stem"], video_sha256=vh, updates=updates,
               mean_loss=float(np.mean(losses)) if losses else None, shard_id=a.shard_id)
    with (out / "progress.jsonl").open("a") as f:
      f.write(json.dumps(row) + "\n")
    torch.save(dict(
      model=head.state_dict(), optimizer=opt.state_dict(), metadata=meta,
      next_source_index=idx + 1, updates=updates, shard_id=a.shard_id, num_shards=a.num_shards,
    ), out / "latest.tmp")
    (out / "latest.tmp").replace(out / "latest.pt")
    print("TRAIN", f"shard{a.shard_id}", idx + 1, len(order), r["stem"], updates, flush=True)

  if any(not torch.isfinite(v).all() for v in head.state_dict().values()):
    raise RuntimeError("Nonfinite reward weights")
  if any(not torch.equal(v.detach().cpu(), bc["model"][k]) for k, v in policy.state_dict().items()):
    raise RuntimeError("Frozen BC changed")
  torch.save(dict(model=head.state_dict(), metadata=meta), out / "reward.pt")
  (out / "summary.json").write_text(json.dumps(dict(
    status="complete", updates=updates, failed_train_sources=failures,
    shard_id=a.shard_id, num_shards=a.num_shards, shard_sources=shard_seen,
    resume_start_source=start_source, rl_ready=False), indent=2))
  # Calibration only on shard 0 to avoid duplicate work / overlap checks.
  if a.shard_id == 0 and a.num_shards == 1:
    scores = []
    head.eval()
    for r in cal:
      try:
        items, vh = prepare(r)
      except (ValueError, OSError) as exc:
        with (out / "failed_calibration_sources.jsonl").open("a") as f:
          f.write(json.dumps(dict(stem=r["stem"], error=str(exc))) + "\n")
        continue
      if vh in seen_video_hashes:
        with (out / "failed_calibration_sources.jsonl").open("a") as f:
          f.write(json.dumps(dict(stem=r["stem"], error="Video content overlaps training")) + "\n")
        continue
      with torch.no_grad():
        for h, n, t, y, m, tid in items:
          z = head(h, n, t)
          for k in y:
            for i in torch.where(m[k][0])[0].tolist():
              scores.append(dict(
                task=tid, head=k, score=float(z[k][0, i].sigmoid()),
                label=int(y[k][0, i]), session=r["source_session"].rsplit("-", 1)[-1]))
    (out / "calibration_scores.json").write_text(json.dumps(scores))
    (out / "calibration.json").write_text(json.dumps(
      calibration_with_confirmation(scores, 5, [r["source_session"].rsplit("-", 1)[-1] for r in cal]), indent=2))


if __name__ == "__main__":
  main()
