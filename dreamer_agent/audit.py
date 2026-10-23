"""Read-only local P0 inventory. Never treats asset presence as capability."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def audit(root: Path) -> dict:
    import torch
    plan = root / 'docs/PLAN_V6.md'
    weights = root / 'checkpoints'
    files = sorted(str(p.relative_to(root)) for p in weights.rglob('*') if p.is_file()) if weights.exists() else []
    return {
        'schema_version': 1,
        'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'checkpoint_files': files,
        'scope': 'root checkpoints only; other locations not searched',
        'p0_passed': False,
        'pending': ['verified weight revisions', 'action round-trip validation',
                    'real-context inference', 'cache branch isolation',
                    'semantic conservation tests', 'throughput and memory measurement'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve()), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
