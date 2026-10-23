"""Content-verified offline RL inputs. Tensor-only training inputs are rejected."""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from .provenance import sha256
from .contracts import Episode, validate_dataset


@dataclass
class OfflineBatch:
    history: torch.Tensor
    tasks: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    episodes: tuple
    binding_sha256: str

    def validate(self, expected):
        validate_dataset(list(self.episodes))
        if not self.episodes or any(e.split != 'train' for e in self.episodes):
            raise ValueError('BC batches must contain offline train episodes')
        if not expected or self.binding_sha256 != expected:
            raise ValueError('BC data binding mismatch')
        if len(self.episodes) != self.history.shape[0]:
            raise ValueError('One source record per batch sequence required')
        if not torch.isfinite(self.history).all():
            raise ValueError('Nonfinite BC history')
        return self.history, self.tasks, self.targets, self.mask


def load_bc_batch(manifest_path, expected_hash, device='cpu'):
    """One reviewed batch per manifest. Safe tensor loading, original source hashes.

    Each sequence has its own Episode provenance. Feature extractor and frozen
    backbone hashes are part of the manifest whose hash is bound by the gate.
    """
    p = Path(manifest_path).resolve()
    if sha256(p) != expected_hash:
        raise ValueError('BC manifest changed')
    m = json.loads(p.read_text())
    if m.get('schema_version') != 1 or not m.get('feature_contract_sha256'):
        raise ValueError('Versioned feature provenance required')
    for entry in m['sources'] + [m['tensors']]:
        if sha256(p.parent / entry['path']) != entry['sha256']:
            raise ValueError('BC source content changed')
    data = torch.load(p.parent / m['tensors']['path'], map_location=device, weights_only=True)
    batch = OfflineBatch(*(data[k] for k in ('history','tasks','targets','mask')),
        tuple(Episode(**e) for e in m['episodes']), expected_hash)
    batch.validate(expected_hash)
    return batch
