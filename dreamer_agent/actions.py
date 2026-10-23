"""Validated local-factor <-> WM token mapping, without legacy training code."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    slots: tuple[tuple[int, ...], ...]
    codec_sha256: str
    frame_hz: float
    action_repeat: int

    def __post_init__(self):
        if not self.slots or not any(len(s) > 1 for s in self.slots) or any(not s or len(set(s)) != len(s) for s in self.slots):
            raise ValueError("Every slot needs a nonempty, unique support set")
        if any(type(v) is not int or v < 0 for s in self.slots for v in s):
            raise ValueError("Token IDs must be nonnegative integers")
        if len(self.codec_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.codec_sha256):
            raise ValueError("Codec content hash required")
        if not np.isfinite(self.frame_hz) or self.frame_hz <= 0 or type(self.action_repeat) is not int or self.action_repeat < 1:
            raise ValueError("Explicit action timebase required")

    @property
    def action_sizes(self):
        return tuple(len(s) for s in self.slots if len(s) > 1)

    def to_dict(self):
        return dict(slots=[list(s) for s in self.slots], codec_sha256=self.codec_sha256,
                    frame_hz=self.frame_hz, action_repeat=self.action_repeat)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(tuple(s) for s in data["slots"]), data["codec_sha256"],
                   data["frame_hz"], data["action_repeat"])

    def to_local(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim < 1 or tokens.shape[-1] != len(self.slots) or tokens.dtype.kind not in "iu":
            raise ValueError("Expected integer WM slots")
        factors = []
        for i, support in enumerate(self.slots):
            values = tokens[..., i]
            if not np.isin(values, support).all():
                raise ValueError(f"Illegal token in slot {i}")
            if len(support) > 1:
                factors.append(np.argmax(values[..., None] == np.array(support), axis=-1))
        return np.stack(factors, axis=-1)

    def to_tokens(self, local):
        local = np.asarray(local)
        if local.ndim < 1 or local.shape[-1] != len(self.action_sizes) or local.dtype.kind not in "iu":
            raise ValueError("Expected integer local action factors")
        columns, k = [], 0
        for support in self.slots:
            if len(support) == 1:
                columns.append(np.full(local.shape[:-1], support[0], dtype=np.int64))
            else:
                ids = local[..., k]
                if ((ids < 0) | (ids >= len(support))).any():
                    raise ValueError("Local action outside support")
                columns.append(np.asarray(support)[ids])
                k += 1
        return np.stack(columns, axis=-1)
