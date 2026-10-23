import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_tool", Path(__file__).resolve().parents[1] / "scripts/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "LICENSE").write_text("Example license")
        (self.root / "README.md").write_text("Example project")
        self.manifest(["LICENSE", "README.md", "release_manifest.json"])

    def manifest(self, files):
        (self.root / "release_manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))

    def test_export_selects_only_reviewed_files(self):
        (self.root / "private.txt").write_text("private")
        payload, issues = release.inspect(self.root)
        self.assertEqual(issues, [])
        output = self.root / "release.tar.gz"
        release.export(payload, output)
        with tarfile.open(output) as archive:
            self.assertEqual(set(archive.getnames()), {"dreamer-agent/" + name for name in [*payload, "SHA256SUMS"]})
        with self.assertRaises(FileExistsError):
            release.export(payload, output)

    def test_symlink_rejected(self):
        (self.root / "README.md").unlink()
        (self.root / "README.md").symlink_to(self.root / "LICENSE")
        with self.assertRaises(ValueError):
            release.inspect(self.root)

    def test_parent_traversal_rejected(self):
        self.manifest(["LICENSE", "README.md", "release_manifest.json", "../private"])
        with self.assertRaises(ValueError):
            release.inspect(self.root)

    def test_secret_report_omits_secret(self):
        secret = "ghp_" + "a" * 36
        (self.root / "README.md").write_text(secret)
        _, issues = release.inspect(self.root)
        self.assertEqual(issues, [{"path": "README.md", "reason": "github-token"}])
        self.assertNotIn(secret, json.dumps(issues))

    def test_missing_file_rejected(self):
        (self.root / "LICENSE").unlink()
        with self.assertRaises(ValueError):
            release.inspect(self.root)

    def test_binary_rejected(self):
        (self.root / "README.md").write_bytes(b"\x00")
        _, issues = release.inspect(self.root)
        self.assertEqual(issues[0]["reason"], "binary")


if __name__ == "__main__":
    unittest.main()
