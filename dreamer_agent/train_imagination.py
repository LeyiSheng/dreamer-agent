"""Guarded P2 runner. --check verifies evidence without importing a backend.

The explicit integration factory returns (trainer, roots, offline_batches).
Roots yield (root,task,seed); offline batches contain only verified train data.
No default old checkpoint, guessed WM frame alignment, or permissive gate.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time
import torch
from .provenance import sha256
from .imagination import RLConfig

REQUIRED = ('world_model_semantics', 'reward_calibration', 'real_bc_baseline', 'integration_contract')


def check_evidence(path, config):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get('schema_version') != 1 or doc.get('stage') != 'P2':
        raise ValueError('Expected versioned P2 evidence manifest')
    if doc.get('config') != asdict(config):
        raise ValueError('Full RL configuration is not approved')
    if doc.get('timebase') != {'frame_hz': 20, 'action_repeat': 1}:
        raise ValueError('Reviewed 20Hz timebase required')
    if not doc.get('seeds') or not doc.get('tasks'):
        raise ValueError('Explicit experiment seeds and tasks required')
    if config.horizon not in doc.get('approved_horizons', []):
        raise ValueError('Horizon has not been reviewed')
    bindings = doc.get('bindings', {})
    required_bindings = ('bc', 'reward', 'tokenizer', 'dynamics', 'action_spec', 'feature_spec', 'plan', 'integration_code', 'training_data', 'roots')
    if any(not bindings.get(k) for k in required_bindings):
        raise ValueError('Missing immutable artifact bindings')
    for name in required_bindings:
        entry = bindings[name]
        if sha256(path.parent / entry['path']) != entry['sha256']:
            raise ValueError('Artifact hash mismatch: '+name)
    for name in REQUIRED:
        entry = doc.get('evidence', {}).get(name, {})
        report_path = path.parent / entry.get('path', '__missing__')
        if not report_path.is_file() or sha256(report_path) != entry.get('sha256'):
            raise ValueError('Missing/changed evidence: '+name)
        report = json.loads(report_path.read_text())
        if report.get('status') != 'accepted' or report.get('kind') != name or not report.get('reviewer'):
            raise ValueError('Evidence needs explicit review: '+name)
        if report.get('artifact_hashes') != {k: bindings[k]['sha256'] for k in required_bindings}:
            raise ValueError('Evidence and current artifacts disagree: '+name)
        if report.get('synthetic') is not False:
            raise ValueError('Synthetic tests cannot unlock production RL')
        if not report.get('raw_records'):
            raise ValueError('Raw evidence records required: '+name)
        for raw in report['raw_records']:
            if sha256(report_path.parent / raw['path']) != raw['sha256']:
                raise ValueError('Raw evidence changed: '+name)
        if name == 'reward_calibration':
            entries = report.get('tasks', {})
            for task in doc['tasks']:
                r = entries.get(str(task), {})
                if not (0 <= r.get('threshold', -1) <= 1 and r.get('confirmation_precision_lower', 0) >= .95
                        and r.get('confirmation_predictions', 0) >= 73 and r.get('background_evaluated') is True
                        and r.get('independent_confirmation') is True):
                    raise ValueError('Reward confirmation insufficient for task '+str(task))
            if report.get('thresholds') != doc.get('thresholds'):
                raise ValueError('Runtime reward thresholds disagree')
    doc['_manifest_path'] = str(path)
    return doc


def run(trainer, roots, offline_batches, output, updates, evidence):
    """Called only after production gate validation; exclusive output directory."""
    if not isinstance(evidence, dict) or '_manifest_path' not in evidence:
        raise ValueError('Verified evidence manifest required by run')
    evidence = check_evidence(evidence['_manifest_path'], trainer.config)
    if trainer.seed not in evidence['seeds'] or trainer.thresholds != {int(k): v for k,v in evidence['thresholds'].items()}:
        raise ValueError('Runtime seed/threshold mismatch')
    if trainer.data_binding != evidence['bindings']['training_data']['sha256']:
        raise ValueError('Runtime BC data mismatch')
    if getattr(trainer, 'artifact_hashes', None) != {k:v['sha256'] for k,v in evidence['bindings'].items()}:
        raise ValueError('Loaded runtime artifacts are not bound')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(config=asdict(trainer.config), evidence=evidence,
        requested_updates=updates, reward_source='learned_imagination', real_success_measured=False), indent=2))
    started = time.monotonic()
    batches = iter(offline_batches)
    with (out/'metrics.jsonl').open('x') as log:
        for root, task, seed in roots:
            trajectory, metrics = trainer.collect(root, task, seed=seed)
            if trajectory is not None:
                metrics.update(trainer.update(trajectory, next(batches) if trainer.config.bc_weight else None))
            metrics['elapsed_seconds'] = time.monotonic() - started
            log.write(json.dumps(metrics)+'\n'); log.flush()
            if trajectory is not None:
                checkpoint = dict(actor=trainer.actor.state_dict(), value=trainer.value.state_dict(),
                    prior=trainer.prior.state_dict(), optimizer=trainer.optimizer.state_dict(),
                    generator=trainer.generator.get_state(), update=trainer.version,
                    config=asdict(trainer.config), evidence=evidence, real_success_measured=False)
                torch.save(checkpoint, out/'latest.tmp')
                (out/'latest.tmp').replace(out/'latest.pt')
            if trainer.version >= updates:
                break
    status = 'complete' if trainer.version >= updates else 'roots_exhausted'
    (out/'summary.json').write_text(json.dumps(dict(status=status, updates=trainer.version,
        real_success_measured=False), indent=2))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    p.add_argument('--factory', help='Explicit reviewed module:function integration')
    p.add_argument('--output', type=Path)
    p.add_argument('--updates', type=int, default=100)
    a = p.parse_args()
    config = RLConfig(**json.loads(a.config.read_text()))
    evidence = check_evidence(a.evidence, config)
    if a.check:
        print('Evidence accepted; no model loaded and no training started')
        return
    if not a.factory or a.output is None or a.updates < 1:
        p.error('Training requires --factory, --output, and positive --updates')
    module, function = a.factory.split(':')
    spec = importlib.util.find_spec(module)
    if spec is None or sha256(spec.origin) != evidence['bindings']['integration_code']['sha256']:
        raise ValueError('Integration factory code is not the reviewed artifact')
    trainer, roots, batches = getattr(importlib.import_module(module), function)(config, evidence)
    if trainer.config != config:
        raise ValueError('Factory changed reviewed configuration')
    run(trainer, roots, batches, a.output, a.updates, evidence)


if __name__ == '__main__':
    main()
