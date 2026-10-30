import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from verify_models import check, load_manifest
from get_models import install


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'assets'
        self.source.mkdir()
        self.data = b'synthetic checkpoint bytes, not a neural model'
        self.spec = {'filename':'test.pth', 'bytes':len(self.data),
                     'sha256':hashlib.sha256(self.data).hexdigest()}
        (self.source/'test.pth').write_bytes(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def test_install_and_reuse_verified_local_asset(self):
        dest = self.root/'models'
        path = install(self.spec, dest, from_local=self.source)
        check(path, self.spec)
        self.assertEqual(install(self.spec, dest, from_local=self.source), path)

    def test_tampered_asset_never_becomes_final_file(self):
        (self.source/'test.pth').write_bytes(b'x'*len(self.data))
        dest = self.root/'models'
        with self.assertRaises(ValueError):
            install(self.spec, dest, from_local=self.source)
        self.assertFalse((dest/'test.pth').exists())
        self.assertEqual(list(dest.iterdir()), [])

    def test_existing_wrong_file_is_preserved(self):
        dest = self.root/'models'; dest.mkdir()
        target = dest/'test.pth'; target.write_bytes(b'user-owned file')
        with self.assertRaises(ValueError):
            install(self.spec, dest, from_local=self.source)
        self.assertEqual(target.read_bytes(), b'user-owned file')

    def test_manifest_rejects_traversal(self):
        path = self.root/'manifest.json'
        path.write_text(json.dumps({'models':[dict(self.spec, filename='../test.pth')]}))
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_help_does_not_require_neural_libraries(self):
        result = subprocess.run([sys.executable, str(ROOT/'run_pipeline.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--check', result.stdout)


if __name__ == '__main__':
    unittest.main()
