"""Apply identical loader failure handling and early checkpoint protection."""
from pathlib import Path
import sys
source=Path(sys.argv[1])
f=source/'core/dom_distributed.py';t=f.read_text();pos=t.index('\n\ndef leader_call')
helper='''\n\ndef coordinated_batches(loader):
    """Make every rank learn a loader failure before starting GPU backward."""
    iterator = iter(loader)
    while True:
        batch, error, ended = None, None, False
        try:
            batch = next(iterator)
        except StopIteration:
            ended = True
        except Exception as exc:
            error = f'rank {dist.get_rank()}: {type(exc).__name__}: {exc}'
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (error, ended), group=_CONTROL_GROUP)
        failures = [error for error, _ in reports if error]
        if failures:
            raise RuntimeError('Distributed data loading failed before backward:\\n' + '\\n'.join(failures))
        if all(done for _, done in reports):
            return
        if any(done for _, done in reports):
            raise RuntimeError('Distributed loaders ended at different microsteps')
        yield batch
'''
assert 'def coordinated_batches' not in t;t=t[:pos]+helper+t[pos:];f.write_text(t)
f=source/'core/dom_distributed_training.py';t=f.read_text();t=t.replace('from core.dom_stopping import', 'from core.dom_distributed import coordinated_batches\nfrom core.dom_stopping import',1);assert '    for batch in loader:' in t;t=t.replace('    for batch in loader:', '    for batch in coordinated_batches(loader):',1)
old='                    if halt or bool(due):save()';new='''                    # Protect initial training before the first 30-minute save.
                    early_save = progress['updates'] <= 1000 and progress['updates'] % 25 == 0
                    if halt or bool(due) or early_save:save()''';assert old in t;t=t.replace(old,new);f.write_text(t)
f=source/'utils/dom_data.py';t=f.read_text();old='''            if decoder is None:
                decoder = torchcodec.decoders.VideoDecoder(episode[key], seek_mode="approximate", num_ffmpeg_threads=1)
            self.decoders[cache_key] = decoder
            while len(self.decoders) > 8:
                self.decoders.popitem(last=False)
            result[key] = self.base._get_video_frames(decoder, ts).squeeze(0)''';new='''            try:
                if decoder is None:
                    decoder = torchcodec.decoders.VideoDecoder(episode[key], seek_mode="approximate", num_ffmpeg_threads=1)
                self.decoders[cache_key] = decoder
                while len(self.decoders) > 8:
                    self.decoders.popitem(last=False)
                result[key] = self.base._get_video_frames(decoder, ts).squeeze(0)
            except Exception as exc:
                path = self.base.root/self.meta.get_video_file_path(episode["_dom_episode_id"], key)
                raise RuntimeError(f'DOM video decode failed: episode={episode["_dom_episode_id"]} camera={key} path={path}: {exc}') from exc''';assert old in t;t=t.replace(old,new);f.write_text(t)
print('Updated loader coordination, early checkpoint protection and decoding context.')
