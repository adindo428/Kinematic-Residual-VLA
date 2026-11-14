#!/usr/bin/env python3
"""Upstream simulator with collection-consistent auxiliary target observations."""
import logging
import os
from pathlib import Path
import sys


class CollectionObjectSelector:
    """Follow collection's fastest-first, then next-unplaced-object convention."""
    def __init__(self, env, sim):
        self.env, self.sim = env.unwrapped, sim
        assert self.env.num_envs == 1
        term = getattr(self.env.cfg.terminations, 'objects_placed', None)
        self.params = term.params if term is not None else {}
        self.remaining = list(self.params.get('objects', ['object']))
        self.current = self.next_object() if len(self.remaining) > 1 else self.remaining[0]

    def next_object(self):
        i = self.sim.get_next_object([self.remaining], self.env.scene)[0]
        name = self.remaining[i]
        logging.info('Collection auxiliary target: %s; remaining %s', name, self.remaining)
        return name

    def state(self):
        if len(self.remaining) > 1:
            scene = self.env.scene
            robot = scene['robot'].data
            current = self.sim.get_curr_state(
                ee_state=None, object_state=scene[self.current].data,
                object_size=self.params['object_sizes'][self.current],
                container_state=scene['container'].data,
                container_size=self.params['container_size'],
                env_origins=robot.root_pos_w, robot_quat=robot.root_quat_w,
                device=self.env.device)
            if self.sim.helpers.is_object_placed(current['object']['pos'], current['object']['size'],
                                                 current['container']['pos'], current['container']['size']).any():
                self.remaining.remove(self.current)
                self.current = self.next_object()
        return self.env.scene[self.current].data


def main():
    scripts = Path(__file__).resolve().parents[1] / 'evaluation/official_dom/scripts'
    sys.path.insert(0, str(scripts))
    from server_observer import instrument_and_run
    source = Path(os.environ['DOM_OFFICIAL_SOURCE']) / 'simulations/evaluate.py'
    text = source.read_text()
    replacements = {
        '    rcv_action = False': '    rcv_action = False\n    _collection_target = _CollectionObjectSelector(env, sim)',
        '            object_state=env.unwrapped.scene["object"].data,': '            object_state=_collection_target.state(),',
        '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},':
            '                "observation.environment_state": {"object": {\n'
            '                    k: v.cpu().numpy() for k, v in curr_state["object"].items()}},\n'
            '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},',
        '    main(app_launcher.app, args)': '    _instrument_and_run(main, app_launcher.app, args, globals())'}
    for before, after in replacements.items():
        assert text.count(before) == 1, 'Upstream simulator hook changed: ' + before
        text = text.replace(before, after)
    sys.path.insert(0, str(source.parent.parent))
    namespace = {'__name__': '__main__', '__file__': str(source), '_instrument_and_run': instrument_and_run,
                 '_CollectionObjectSelector': CollectionObjectSelector}
    exec(compile(text, str(source), 'exec'), namespace)


if __name__ == '__main__':
    main()
