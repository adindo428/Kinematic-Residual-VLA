"""Resume the historical nine-category suite after the completed noise sweep."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

import extend_dom_krv_scene_first as continuation

ROOT = continuation.old.ROOT/'scene-first-resume-20261117'


class CheckedStop(threading.Event):
    def wait(self, timeout=None):
        stopped = super().wait(timeout)
        log = ROOT/'execution/server_shared.log'
        if not stopped and log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-65536))
                if b'Traceback (most recent call last):' in stream.read():
                    raise RuntimeError('Full-suite simulator exception; inspect execution/server_shared.log')
        return stopped


def main():
    noise_root = continuation.low.WORK/'evaluations/velocity-uncertainty-cr200-20261116'
    result = json.loads((noise_root/'completion.json').read_text())
    assert result['verified'] and result['trials'] == 1000
    assert not (ROOT/'plan.json').exists(), 'Do not overwrite an existing continuation'
    ROOT.mkdir(exist_ok=True, parents=True)
    (ROOT/'request.json').write_text(json.dumps(dict(
        requested_utc=datetime.now(timezone.utc).isoformat(), target_trials_per_model=1800,
        source_pause_manifest=str(continuation.old.ROOT/'vu200-pause.json'),
        completed_noise_sweep=str(noise_root/'completion.json'),
        scope='Original DOM raw-action baseline and KRV epoch4+partial5 constant gate; no VPD/VI/noise/tracker',
        per_category_target=200, categories=['CR','DA','LS','VU','SR','MP','VG','MG','DR'],
        historical_evaluator_preserved=True,
        note='The separate noise sweep used DOMV. This resumes the original paused DOM/KRV nine-category suite, retaining original reset/streaming semantics.'), indent=2)+'\n')
    continuation.ROOT = ROOT
    continuation.runner.STOP = CheckedStop()
    continuation.main()


if __name__ == '__main__':
    main()
