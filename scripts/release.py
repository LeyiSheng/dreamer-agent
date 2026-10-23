"""Inspect and export explicitly selected public source files (standard library only)."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

MAX_BYTES = 2 * 1024 * 1024
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "credential-literal": re.compile(r'''(?im)^\s*(?:["']?)(?:password|api_key|access_token|client_secret)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=.-]{16,}["']'''),
    "private-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/|/data_sd[a-z]/"),
    "private-host": re.compile(r"\bhkustgz_inside\b|\b10\.120\.17\.96\b"),
}


def collect(root):
    manifest = json.loads((root / "release_manifest.json").read_text())
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, list):
        raise ValueError("Invalid release manifest")
    if any(not isinstance(x, str) for x in files) or len(files) != len(set(files)):
        raise ValueError("Manifest entries must be unique strings")
    for required in ("LICENSE", "README.md", "release_manifest.json"):
        if required not in files:
            raise ValueError("Missing required entry: " + required)
    for name in files:
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts or str(rel) != name or "\\" in name:
            raise ValueError("Unsafe manifest path")
        path = root
        for part in rel.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("Symlink in manifest: " + name)
            if part == ".git" or (path.is_dir() and (path / ".git").exists()):
                raise ValueError("Nested repository: " + name)
        if not path.is_file():
            raise ValueError("Missing file: " + name)
    return sorted(files)


def inspect(root):
    files = collect(root)
    issues, payload = [], {}
    for name in files:
        path = root / name
        if path.stat().st_size > MAX_BYTES:
            issues.append({"path": name, "reason": "oversize"})
            continue
        data = path.read_bytes()
        payload[name] = data
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append({"path": name, "reason": "binary"})
            continue
        if "\x00" in content:
            issues.append({"path": name, "reason": "binary"})
        for reason, pattern in PATTERNS.items():
            if pattern.search(content):
                issues.append({"path": name, "reason": reason})
    # Archives have no Git metadata. Workspaces get an additional tracked-file check.
    if (root / ".git").exists():
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 capture_output=True, check=True).stdout
        for name in tracked.decode().split("\0"):
            if name and name not in files:
                issues.append({"path": name, "reason": "tracked-outside-manifest"})
    return payload, issues


def export(payload, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    sums = "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n"
                   for name, data in sorted(payload.items())).encode()
    members = dict(payload, **{"SHA256SUMS": sums})
    # Exclusive creation protects existing release artifacts.
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo("dreamer-agent/" + name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "export"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload, issues = inspect(args.root.resolve())
        print(json.dumps({"files": len(payload), "issues": issues}, ensure_ascii=False, indent=2))
        if issues:
            return 1
        if args.command == "export":
            if args.output is None:
                raise ValueError("export requires --output")
            export(payload, args.output)
            print("Exported: " + str(args.output))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("Release check failed: " + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
