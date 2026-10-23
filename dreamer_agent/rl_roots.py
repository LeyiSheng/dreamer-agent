"""Offline root content and tensor binding checks."""
import hashlib
import json
from pathlib import Path
from .provenance import sha256


def tensor_hash(tensor):
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def verify_root(root):
    p = root.provenance
    path = Path(p['manifest_path']).resolve()
    if sha256(path) != p['manifest_sha256']:
        raise ValueError('Root manifest changed')
    m = json.loads(path.read_text())
    if m.get('schema_version') != 1 or m.get('episode') != vars(root.episode):
        raise ValueError('Root episode binding mismatch')
    if root.episode.origin != 'offline' or root.episode.split != 'train':
        raise ValueError('Root is not offline training data')
    if m.get('task_start_frame') != m.get('frame') or type(m.get('frame')) is not int or m['frame'] < 0:
        raise ValueError('Explicit task-start frame required')
    if not m.get('feature_contract_sha256') or not m.get('alignment_evidence_sha256'):
        raise ValueError('Feature and WM alignment contracts required')
    for entry in m['sources']:
        if sha256(path.parent / entry['path']) != entry['sha256']:
            raise ValueError('Root source changed')
    if not m['sources']:
        raise ValueError('Original sources required')
    for key in ('pixels','frame_actions','features','past_actions'):
        if tensor_hash(getattr(root, key)) != m['tensors'][key]:
            raise ValueError('Root tensor changed: '+key)
    return m
