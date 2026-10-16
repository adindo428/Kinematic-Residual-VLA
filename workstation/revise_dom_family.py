#!/usr/bin/env python3
"""Revise a frozen DOM family after approved exclusions, reusing its artifacts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-family', type=Path, required=True)
    parser.add_argument('--family-root', type=Path, required=True)
    parser.add_argument('--approved-exclusions', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import pyarrow.parquet as pq
    from torchcodec.decoders import VideoDecoder
    from core.dom_training import load_family
    from utils.dom_experiment import atomic_json, digest, file_digest, grouped_stratified_split, source_digest
    spec = importlib.util.spec_from_file_location('frozen_preparation', args.source/'scripts/prepare_dom100.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    old, out = args.prior_family.resolve(), args.family_root.resolve()
    prior = load_family(old)
    if prior['source_sha256'] != source_digest(args.source):
        raise ValueError('Numerical source differs from the prior frozen family')
    if (out/'manifest.json').exists():
        raise FileExistsError('The revised family is already frozen')
    approval = json.loads(args.approved_exclusions.read_text())
    if not approval.get('user_authorized_exclusions') or approval['dataset_revision'] != prior['dataset_revision']:
        raise ValueError('Explicit pinned-release exclusion approval is required')
    data = Path(prior['dataset_root'])
    info = json.loads((data/'meta/info.json').read_text())
    previous_selection = json.loads((old/'dataset_selection.json').read_text())
    previous_excluded = {r['episode_index']:r for r in previous_selection['excluded']}
    old_records = json.loads((old/'episodes.json').read_text())
    by_id = {r['episode_index']:r for r in old_records}
    excluded = []
    for entry in approval['episodes']:
        ep = entry['episode_index']
        record = by_id.get(ep, previous_excluded.get(ep))
        if record is None or record['length'] != entry['frames']:
            raise ValueError(f'Unknown or inconsistent excluded episode {ep}')
        for video in entry['invalid_videos']:
            expected_paths = {str(p.relative_to(data)) for p in prep.episode_paths(data, info, ep)[1:]}
            if video['path'] not in expected_paths:
                raise ValueError('Exclusion names an unrelated camera/episode')
            path = data/video['path']
            if path.stat().st_size != video['bytes'] or file_digest(path) != video['sha256']:
                raise ValueError(f'Approved video changed: {path}')
            if 'decoded_frames' in video:
                decoder = VideoDecoder(path, seek_mode='exact', num_ffmpeg_threads=1)
                if len(decoder) != video['decoded_frames'] or not 0 < len(decoder) < record['length']:
                    raise ValueError(f'Short-video evidence differs: {path}')
            else:
                try:
                    VideoDecoder(path, seek_mode='exact', num_ffmpeg_threads=1)
                except RuntimeError:
                    pass
                else:
                    raise ValueError(f'Approved empty video is now decodable: {path}')
        excluded.append({**record, 'missing_or_empty_files': [], 'invalid_video_files': entry['invalid_videos'],
                         'reason': approval['reason']})
    ids = {r['episode_index'] for r in excluded}
    if len(ids) != len(excluded) or not set(previous_excluded).issubset(ids):
        raise ValueError('Duplicate exclusions or a previous exclusion was removed')
    records = [r for r in old_records if r['episode_index'] not in ids]
    if {r['episode_index'] for r in records} != set(range(info['total_episodes']))-ids:
        raise ValueError('The revised selection does not cover all remaining episodes')
    coverage = prep.coverage_summary(records, excluded, info['total_episodes'], info['total_frames'])
    if (coverage['excluded_frames'] != approval['excluded_frames'] or
            coverage['retained_episodes'] != approval['retained_episodes'] or
            coverage['retained_frames'] != approval['retained_frames']):
        raise ValueError('Coverage differs from the approved selection')
    out.mkdir(parents=True, exist_ok=True)
    (out/'initialization').mkdir(exist_ok=True)
    for name in ('config.json', 'model.safetensors'):
        target = out/'initialization'/name
        if not target.exists():
            os.link(old/'initialization'/name, target)
    if file_digest(out/'initialization/model.safetensors') != prior['weights_sha256']:
        raise ValueError('Initialization weights changed')
    files = json.loads((old/'parquet_hashes.json').read_text())
    phases = json.loads((old/'phase_hashes.json').read_text())
    videos = json.loads((old/'video_hashes.json').read_text())
    retained = {r['episode_index'] for r in records}
    files = {str(ep):files[str(ep)] for ep in sorted(retained)}
    phases = {str(ep):phases[str(ep)] for ep in sorted(retained)}
    removed_paths = {str(p.relative_to(data)) for ep in ids-set(previous_excluded)
                     for p in prep.episode_paths(data, info, ep)[1:]}
    videos = {key:value for key,value in videos.items() if key not in removed_paths}
    (out/'phase_cache').mkdir(exist_ok=True)
    for i, ep in enumerate(sorted(retained)):
        target = out/'phase_cache'/f'{ep:06d}.npz'
        if not target.exists():
            os.link(old/'phase_cache'/target.name, target)
        if i % 10000 == 0:
            print(json.dumps({'phase':'reusing_phase_cache', 'episodes':i+1, 'total':len(records)}), flush=True)
    split = grouped_stratified_split(records)
    # Recalculate training-only statistics once for the revised grouped split.
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for i, ep in enumerate(split['train']):
        path = data/info['data_path'].format(episode_chunk=ep//info['chunks_size'], episode_index=ep)
        if file_digest(path) != files[str(ep)]:
            raise ValueError(f'Parquet changed while revising statistics: {ep}')
        values = np.asarray(pq.read_table(path, columns=['observation.environment_state'])[0].to_pylist(),
                            dtype=np.float64)[:, 6:9]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite velocity in episode {ep}')
        sums += values.sum(0)
        squares += np.square(values).sum(0)
        count += len(values)
        if i % 5000 == 0:
            print(json.dumps({'phase':'recalculating_training_statistics', 'episodes':i+1,
                              'total':len(split['train'])}), flush=True)
    by_id = {r['episode_index']:r for r in records}
    if count != sum(by_id[ep]['length'] for ep in split['train']):
        raise ValueError('Training statistics frame count differs')
    selection = {**previous_selection, 'episode_ids':sorted(retained), 'excluded':excluded,
                 'reason':approval['reason']}
    atomic_json(out/'split.json', {'splits':split, 'seed':42, 'ratios':[.98,.01,.01]})
    atomic_json(out/'episodes.json', records)
    atomic_json(out/'pilot_episodes.json', prep.balanced_sample([by_id[e] for e in split['train']],4096))
    total_validation = sum(by_id[e]['length'] for e in split['validation'])
    atomic_json(out/'validation_anchors.json', sorted(np.random.default_rng(42).choice(total_validation,8192,replace=False).tolist()))
    atomic_json(out/'validation_cases.json', prep.balanced_sample([by_id[e] for e in split['validation']],64))
    atomic_json(out/'test_cases.json', prep.balanced_sample([by_id[e] for e in split['test']],256))
    for name, value in [('parquet_hashes.json',files),('phase_hashes.json',phases),('video_hashes.json',videos),
                        ('dataset_selection.json',selection),('dataset_coverage.json',coverage)]:
        atomic_json(out/name, value)
    (out/'approved_exclusions.json').write_bytes(args.approved_exclusions.read_bytes())
    derivation = {'prior_manifest':prior['fingerprint'], 'prior_family':str(old),
                  'numerical_source_sha256':prior['source_sha256'],
                  'revision_adapter_sha256':file_digest(Path(__file__)),
                  'reused_episode_records':len(records), 'reused_phase_files':len(phases),
                  'reused_video_hashes':len(videos), 'newly_excluded_episodes':sorted(ids-set(previous_excluded)),
                  'statistics':'Recalculated from all revised training parquets using the frozen preparation formula'}
    atomic_json(out/'revision_provenance.json', derivation)
    manifest = {**prior, 'initialization':str(out/'initialization'), 'frames':coverage['retained_frames'],
                'episodes':coverage['retained_episodes'], 'coverage':coverage, 'train_frames':count,
                'velocity_mean':(sums/count).tolist(),
                'velocity_std':np.sqrt(np.maximum(squares/count-(sums/count)**2,1e-12)).tolist(),
                'counts':{k:len(v) for k,v in split.items()}, 'split_sha256':file_digest(out/'split.json'),
                'parquets_sha256':digest(files)}
    manifest.pop('fingerprint')
    manifest['artifacts'] = {name:file_digest(out/name) for name in
                             [*prior['artifacts'], 'revision_provenance.json']}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(out/'manifest.json', manifest)
    print(json.dumps({'phase':'revised_family_complete','manifest':manifest['fingerprint'],
                      'episodes':manifest['episodes'],'frames':manifest['frames'],'counts':manifest['counts']}),flush=True)


if __name__ == '__main__':
    main()
