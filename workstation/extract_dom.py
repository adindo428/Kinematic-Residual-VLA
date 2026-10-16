#!/usr/bin/env python3
"""Extract official DOM releases without overwriting conflicting local assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if '__MACOSX' in relative.parts or relative.name.startswith('._'):
                continue
            target = destination / relative
            if destination.resolve() not in target.resolve().parents:
                raise ValueError(f'Archive path leaves destination: {member.filename}')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with target.open('rb') as existing, bundle.open(member) as incoming:
                    if digest(existing) != digest(incoming):
                        raise FileExistsError(f'Conflicting asset; preserve and inspect: {target}')
            else:
                temporary = target.with_name(target.name + '.extracting')
                with bundle.open(member) as incoming, temporary.open('wb') as output:
                    shutil.copyfileobj(incoming, output, 8 * 1024 * 1024)
                temporary.replace(target)
            count += 1
    if archive.name == 'DOM-Test.zip':
        retained = destination / archive.name
        if not retained.exists():
            retained.symlink_to(archive.resolve())
    return {'archive': str(archive.resolve()), 'destination': str(destination.resolve()),
            'files_verified_or_extracted': count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([extract(path, args.assets) for path in args.archives], indent=2))
