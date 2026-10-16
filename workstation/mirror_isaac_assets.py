#!/usr/bin/env python3
"""Mirror required NVIDIA assets byte-for-byte, retaining their relative paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

from pxr import UsdUtils

SOURCE = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/'
SEEDS = [
    'Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd',
    'Isaac/Environments/Grid/default_environment.usd',
    'Isaac/Props/UIElements/frame_prim.usd',
    'Isaac/Props/UIElements/arrow_x.usd',
]


def mirror(destination):
    destination.mkdir(parents=True, exist_ok=True)
    queue = [SOURCE + path for path in SEEDS]
    seen = set()
    records = []
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if not url.startswith(SOURCE):
            raise ValueError(f'Asset outside the declared mirror root: {url}')
        relative = Path(unquote(url[len(SOURCE):]))
        path = destination / relative
        if destination.resolve() not in path.resolve().parents:
            raise ValueError(f'Asset leaves mirror directory: {url}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name + '.downloading')
            with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            temporary.replace(path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        records.append({'source': url, 'path': str(path.resolve()), 'sha256': h.hexdigest(),
                        'bytes': path.stat().st_size})
        print(f'Mirrored {relative}', flush=True)
        if path.suffix.lower() in ('.usd', '.usda', '.usdc'):
            for group in UsdUtils.ExtractExternalReferences(str(path)):
                for reference in group:
                    # Built-in MDL shaders resolve through the installed Kit runtime.
                    if reference.endswith('.mdl') and '/' not in reference:
                        continue
                    queue.append(urljoin(url, reference))
    report = {'complete': True, 'source_root': SOURCE, 'files': records,
              'content_modified': False,
              'note': '5.0 is the asset root declared by the unmodified Isaac Lab v2.2.1 rendering experience, including when running Sim 4.5.'}
    (destination / 'mirror.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    mirror(args.destination)
