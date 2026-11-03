"""Apply an explicit stopping cap to two approved jobs without rebinding checkpoints.

The original recipe remains the optimizer/scheduler/checkpoint identity. Only the
training loop's stopping limit changes. Readiness and DataLoader math are intact.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

CONTROL_PATH = os.environ.get('DVLA_EPOCH_CONTROL_FILE')
BOOTSTRAP_ACTIVE = False
TRAINING_CONTROL_INSTALLED = False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_control():
    path = Path(CONTROL_PATH)
    value = json.loads(path.read_text())
    assert digest({k: v for k, v in value.items() if k != 'fingerprint'}) == value['fingerprint']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == value['bootstrap_sha256']
    assert value['epochs'] == 10 and value['scheduler_horizon_epochs'] == 100
    job = os.environ.get('SLURM_JOB_ID')
    assert job in value['jobs'], 'Stopping control is scoped to two existing job IDs'
    return value, value['jobs'][job]


def operational_control(root, variant, recipe, manifest):
    value, job = read_control()
    assert Path(root).resolve() == Path(job['family']).resolve()
    assert variant == job['variant']
    assert manifest['fingerprint'] == job['manifest']
    assert manifest['source_sha256'] == value['source_sha256']
    assert recipe['epochs'] == value['scheduler_horizon_epochs']
    assert recipe['effective_batch'] == 352 and recipe['lr'] == 3e-5
    # recipe.json is allowed to acquire its actual runtime approval before the
    # first KRVLA allocation. Verify its own frozen fingerprint either way.
    assert digest({k: v for k, v in recipe.items() if k != 'fingerprint'}) == recipe['fingerprint']
    import torch.distributed as dist
    assert dist.get_world_size() == job['world_size']
    return {'epochs': value['epochs'], 'scheduler_horizon_epochs': value['scheduler_horizon_epochs'],
            'fingerprint': value['fingerprint'], 'control_file': str(CONTROL_PATH),
            'authorization': value['authorization'], 'checkpoint_identity': 'original frozen recipe retained',
            'scope': 'total completed epochs, including restored progress'}


def install_training_control(module):
    global TRAINING_CONTROL_INSTALLED
    value, job = read_control()
    source_path = Path(module.__file__).resolve()
    assert str(source_path) == value['trainer_path']
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == value['trainer_sha256']
    assert os.environ.get('DOM100_NODE_MODE') == 'train'
    assert Path(os.environ['DOM100_ROOT']).resolve() == Path(job['family']).resolve()
    original = inspect.getsource(module.train_distributed)
    replacements = [
        ("    max_epochs = recipe['epochs']\n",
         "    epoch_control = _operational_epoch_control(root, variant, recipe, manifest)\n"
         "    max_epochs = epoch_control['epochs']\n"),
        ("        policy.config.scheduler_decay_steps = max_epochs*math.ceil(len(data)/352)\n",
         "        policy.config.scheduler_decay_steps = epoch_control['scheduler_horizon_epochs']*math.ceil(len(data)/352)\n"),
        ("        writer = None\n",
         "        progress['operational_epoch_cap'] = epoch_control['epochs']\n"
         "        progress['scheduler_horizon_epochs'] = epoch_control['scheduler_horizon_epochs']\n"
         "        leader_call(lambda:atomic_json(destination/'operational-control.json',epoch_control))\n"
         "        logging.info('%s total epoch cap=%d; original scheduler horizon=%d; restored updates=%d',"
         "variant,max_epochs,epoch_control['scheduler_horizon_epochs'],progress['updates'])\n"
         "        writer = None\n"),
        ("config={'manifest':manifest,'recipe':recipe,'variant':variant,'runtime':runtime,'world_size':dist.get_world_size()})",
         "config={'manifest':manifest,'recipe':recipe,'variant':variant,'runtime':runtime,'world_size':dist.get_world_size(),"
         "'operational_epoch_cap':epoch_control['epochs'],'operational_training_control':epoch_control})"),
        ("'world_size':dist.get_world_size(),'backend':'DDP/NCCL'})",
         "'world_size':dist.get_world_size(),'backend':'DDP/NCCL','operational_training_control':epoch_control})"),
    ]
    transformed = original
    for old, new in replacements:
        assert transformed.count(old) == 1, 'Unexpected frozen trainer layout: ' + old
        transformed = transformed.replace(old, new)
    ast.parse(transformed)
    module.__dict__['_operational_epoch_control'] = operational_control
    exec(compile(transformed, str(Path(__file__).with_name('controlled_train_distributed.py')), 'exec'), module.__dict__)
    module.train_distributed.__operational_epoch_cap__ = value['epochs']
    module.train_distributed.__scheduler_horizon_epochs__ = value['scheduler_horizon_epochs']
    module.train_distributed.__original_source_sha256__ = value['trainer_sha256']
    TRAINING_CONTROL_INSTALLED = True


class ControlledLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install_training_control(module)


class ControlledFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'core.dom_distributed_training':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None
        spec.loader = ControlledLoader(spec.loader)
        return spec


if CONTROL_PATH:
    try:
        read_control()
        BOOTSTRAP_ACTIVE = True
        if os.environ.get('DOM100_NODE_MODE') == 'train':
            # An import failure terminates the worker; it never falls back to
            # an uncapped trainer after a failed control installation.
            assert 'core.dom_distributed_training' not in sys.modules
            sys.meta_path.insert(0, ControlledFinder())
    except BaseException:
        traceback.print_exc()
        os._exit(78)
