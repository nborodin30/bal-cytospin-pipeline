#!/usr/bin/env python3
"""Download public GitHub Release weights and check the pinned local manifest."""
import argparse
from pathlib import Path
import os
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

from verify_models import ROOT, check, load_manifest


def install(spec, destination, *, repo=None, tag='thesis-v1.0', from_local=None):
    destination = Path(destination)
    target = destination / spec['filename']
    if target.exists():
        # Do not replace an unrelated or corrupted existing file silently.
        check(target, spec)
        print(f'Already verified: {target.name}')
        return target
    if from_local is None and (not repo or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo)):
        raise ValueError('Supply --repo OWNER/REPO for a public release, or --from-local DIRECTORY.')
    destination.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+spec['filename']+'.', suffix='.download', dir=destination)
    temp = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as output:
            if from_local is not None:
                with (Path(from_local) / spec['filename']).open('rb') as source:
                    shutil.copyfileobj(source, output, length=8*1024**2)
            else:
                url = f'https://github.com/{repo}/releases/download/{quote(tag,safe="")}/{quote(spec["filename"],safe="")}'
                request = Request(url, headers={'User-Agent': 'BAL-cytospin-pipeline/1.0'})
                with urlopen(request, timeout=120) as source:
                    shutil.copyfileobj(source, output, length=8*1024**2)
        check(temp, spec)
        if target.exists():
            raise FileExistsError(f'Another process created {target}; nothing was replaced.')
        # Link is atomic and refuses to overwrite a file created concurrently.
        try:
            os.link(temp, target)
        except OSError as e:
            # exFAT and other filesystems may not support hard links.
            if isinstance(e, FileExistsError):
                raise
            created = False
            try:
                with target.open('xb') as output, temp.open('rb') as source:
                    created = True
                    shutil.copyfileobj(source, output, length=8*1024**2)
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                raise
        print(f'Installed and verified: {target.name}')
        return target
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', help='Actual GitHub OWNER/REPO, public releases only')
    p.add_argument('--tag', help='Release tag; defaults to the pinned manifest tag')
    p.add_argument('--from-local', type=Path, help='Install from a folder of downloaded release assets')
    p.add_argument('--models-dir', type=Path, default=ROOT/'models')
    p.add_argument('--manifest', type=Path, default=ROOT/'models/manifest.json')
    args = p.parse_args(argv)
    if bool(args.repo) == bool(args.from_local):
        p.error('Choose exactly one of --repo or --from-local.')
    manifest = load_manifest(args.manifest)
    for spec in manifest['models']:
        install(spec, args.models_dir, repo=args.repo,
                tag=args.tag or manifest['release_tag'], from_local=args.from_local)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
