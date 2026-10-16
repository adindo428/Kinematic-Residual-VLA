#!/usr/bin/env python3
"""Read MP4 sample tables to audit required DOM views without decoding frames."""
import argparse
import json
from pathlib import Path
import struct
import time


def atoms(stream, start, end):
    pos = start
    while pos + 8 <= end:
        stream.seek(pos)
        size, kind = struct.unpack('>I4s', stream.read(8))
        header = 8
        if size == 1:
            size = struct.unpack('>Q', stream.read(8))[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f'Invalid MP4 atom {kind!r} at {pos}')
        yield kind, pos + header, pos + size
        pos += size


def counts(stream, start, end):
    found = []
    for kind, begin, stop in atoms(stream, start, end):
        if kind in (b'moov', b'trak', b'mdia', b'minf', b'stbl'):
            found.extend(counts(stream, begin, stop))
        elif kind == b'stsz':
            stream.seek(begin + 8)
            found.append(struct.unpack('>I', stream.read(4))[0])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--family', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    episodes = json.loads((args.family / 'episodes.json').read_text())
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    cameras = ['observation.images.opst_cam', 'observation.images.wrist_cam']
    if not all(camera in info['features'] for camera in cameras):
        raise ValueError('Required cameras are absent from DOM metadata')
    start = time.monotonic()
    checked = 0
    mismatches = []
    for number, episode in enumerate(episodes):
        ep, expected = episode['episode_index'], episode['length']
        for camera in cameras:
            relative = info['video_path'].format(episode_chunk=ep // info['chunks_size'],
                                                episode_index=ep, video_key=camera)
            path = args.dataset / relative
            try:
                with path.open('rb') as stream:
                    actual = counts(stream, 0, path.stat().st_size)
                if actual != [expected]:
                    mismatches.append(dict(episode=ep, camera=camera, path=relative,
                                           expected=expected, sample_counts=actual))
            except Exception as error:
                mismatches.append(dict(episode=ep, camera=camera, path=relative,
                                       expected=expected, error=str(error)))
            checked += 1
        if number % 5000 == 0:
            print(json.dumps(dict(checked=checked, mismatches=len(mismatches),
                                  seconds=round(time.monotonic()-start, 1))), flush=True)
    report = dict(complete=True, checked=checked, episodes=len(episodes),
                  seconds=time.monotonic()-start, mismatches=mismatches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'mismatches': len(mismatches), 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
