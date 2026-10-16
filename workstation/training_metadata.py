"""Remove metadata unused after the frozen DOM constructor computes statistics."""
import json


def install_metadata_compaction(module):
    original = module.build_dataset

    def build_dataset(*args, **kwargs):
        data = original(*args, **kwargs)
        if data.meta is not data.base.meta or not hasattr(data.base, 'stats'):
            raise ValueError('Unexpected frozen DOM metadata/aggregate layout')
        # The frozen adapter uses episodes_stats only in its constructor's
        # aggregate_stats call. __getitem__, _get_episode and all DOMData paths
        # use the retained aggregate, episode indices, tasks, hashes and info.
        aggregate = data.base.stats
        removed = len(data.meta.episodes_stats)
        data.meta.episodes_stats = {}
        if data.base.stats is not aggregate:
            raise ValueError('Aggregate statistics must remain unchanged')
        print(json.dumps({'phase':'dataset_ready_for_workers',
                          'selected_episodes':len(data.episodes),
                          'unused_episode_statistics_removed':removed,
                          'aggregate_statistics_retained':True}),flush=True)
        return data

    module.build_dataset = build_dataset
