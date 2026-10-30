#!/usr/bin/env python3
"""Verify local release assets without importing PyTorch or unpickling files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    if path.stat().st_size != spec['bytes']:
        raise ValueError(f'Size mismatch: {path.name}')
    if digest(path) != spec['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path.name}')


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    seen = set()
    for spec in data['models']:
        name = spec['filename']
        if not name or name != Path(name).name or name in seen:
            raise ValueError('Manifest contains an invalid or duplicate filename.')
        seen.add(name)
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'models/manifest.json')
    p.add_argument('--models-dir', type=Path, default=ROOT/'models')
    args = p.parse_args(argv)
    manifest = load_manifest(args.manifest)
    for spec in manifest['models']:
        check(args.models_dir / spec['filename'], spec)
        print(f"OK  {spec['filename']}  {spec['bytes']:,} bytes  SHA-256 verified")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
