"""Short-lived, same-boot permission to archive perception frames (no motion)."""
import json
import math
import fcntl
from pathlib import Path


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def recording_active(path, now, boot=None):
    if path is None:
        return True
    try:
        row = json.loads(Path(path).read_text())
        remaining = float(row['expires_monotonic_s']) - now
        return (row['boot_id'] == (boot or boot_id()) and row['active'] is True
                and math.isfinite(remaining) and 0 < remaining <= 2)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def renew(path, active, now):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(active=bool(active), boot_id=boot_id(),
                                   expires_monotonic_s=now + .75)))
    temp.replace(path)


class LeaseOwner:
    """Prevent a second recorder from disabling another recorder's archive lease."""
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise ValueError('Another recorder already owns this recording lease') from exc

    def close(self):
        self.stream.close()
