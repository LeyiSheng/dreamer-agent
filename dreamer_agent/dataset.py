"""Strict new-format feature dataset with original-session isolation.

NPZ files must be newly exported with provenance; legacy PT caches are unsupported.
The format contains visual features, local action IDs, task IDs and source frames.
Inventory/coordinates/events are not accepted as policy input arrays.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .actions import ActionSpec
from .contracts import Episode, validate_dataset
from .provenance import sha256

ARRAYS = {"features", "actions", "tasks", "frames"}


class OfflineWindows(Dataset):
    def __init__(self, manifest_path, split, context=32, targets=8):
        if context < 1 or targets < 1 or split not in {"train", "calibration", "test"}:
            raise ValueError("Invalid window/split")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m.get("schema_version") != 1 or m.get("origin") != "offline":
            raise ValueError("Only versioned offline manifests accepted")
        mode = m.get("policy_mode", "task_conditioned")
        if mode not in {"unconditioned", "task_conditioned"}:
            raise ValueError("Unknown policy mode")
        if mode == "unconditioned" and (m["tasks"] != 1 or m.get("task_names") != []):
            raise ValueError("Unconditioned mode requires no skill vocabulary")
        feature = m["feature_spec"]
        if feature.get("kind") != "frozen_visual" or feature.get("frozen") is not True:
            raise ValueError("Requires frozen visual feature export")
        for key in ("extractor_sha256", "preprocessing_sha256"):
            value = feature[key]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Feature provenance needs SHA256 hashes")
        self.spec = ActionSpec.from_dict(m["action_spec"])
        records = m["episodes"]
        validate_dataset([Episode(r["episode_id"], r["source_episode_id"], r["split"], r["origin"]) for r in records])
        self.split = split
        self.context, self.targets = context, targets
        self.episodes, self.windows = [], []
        # Validate all splits before training, including file identities.
        file_splits = {}
        for record in records:
            fingerprint = record["sha256"]
            if fingerprint in file_splits and file_splits[fingerprint] != record["split"]:
                raise ValueError("Identical feature files cross splits")
            file_splits[fingerprint] = record["split"]
            path = (self.manifest_path.parent / record["path"]).resolve()
            if sha256(path) != fingerprint:
                raise ValueError("Feature file hash mismatch")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != ARRAYS:
                    raise ValueError("Unexpected arrays; privileged inputs are forbidden")
                arrays = {k: archive[k].copy() for k in ARRAYS}
            f, a, tasks, frames = (arrays[k] for k in ("features", "actions", "tasks", "frames"))
            t = len(f)
            if t == 0 or f.shape != (t, feature["dimension"]) or not np.isfinite(f).all():
                raise ValueError("Invalid visual features")
            if a.shape != (t, len(self.spec.action_sizes)) or a.dtype.kind not in "iu":
                raise ValueError("Invalid local actions")
            self.spec.to_tokens(a)  # reject out-of-support labels
            if tasks.shape != (t,) or tasks.dtype.kind not in "iu" or ((tasks < 0) | (tasks >= m["tasks"])).any():
                raise ValueError("Invalid task labels")
            if frames.shape != (t,) or frames.dtype.kind not in "iu" or (np.diff(frames) != self.spec.action_repeat).any():
                raise ValueError("Frame gaps must be split into separate contiguous clips")
            if record["split"] == split:
                index = len(self.episodes)
                self.episodes.append(arrays)
                self.windows.extend((index, start) for start in range(0, t, context))
        if not self.windows:
            raise ValueError("Selected split is empty")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep_id, start = self.windows[index]
        ep = self.episodes[ep_id]
        size, factors = self.context, len(self.spec.action_sizes)
        stop = min(start + size, len(ep["features"]))
        valid_length = stop - start
        features = np.zeros((size, self.manifest["feature_spec"]["dimension"]), dtype=np.float32)
        past = np.zeros((size, sum(self.spec.action_sizes)), dtype=np.float32)
        tasks = np.zeros(size, dtype=np.int64)
        labels = np.zeros((size, self.targets, factors), dtype=np.int64)
        valid = np.zeros((size, self.targets), dtype=bool)
        features[:valid_length] = ep["features"][start:stop]
        tasks[:valid_length] = ep["tasks"][start:stop]
        # Right padding only: GRU never consumes another episode's observations.
        for t in range(valid_length):
            pos = start + t
            if pos > 0:
                offset = 0
                for k, n in enumerate(self.spec.action_sizes):
                    past[t, offset + ep["actions"][pos - 1, k]] = 1
                    offset += n
            for horizon in range(self.targets):
                target = pos + horizon
                # Stop MTP at clip end OR any intervening task switch.
                if target >= len(ep["features"]) or ep["tasks"][target] != ep["tasks"][pos]:
                    break
                labels[t, horizon] = ep["actions"][target]
                valid[t, horizon] = True
        return {k: torch.from_numpy(v) for k, v in dict(features=features, past_actions=past,
                tasks=tasks, targets=labels, valid=valid).items()}
