"""Launch the actual simulator and record its running version; no benchmark assets."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--expected-sim', default='5.1.0')
parser.add_argument('--expected-lab', default='2.3.2')
output_args, _ = parser.parse_known_args()
output_args.output.parent.mkdir(parents=True, exist_ok=True)
output_args.output.write_text(json.dumps({'validated': False, 'status': 'starting simulator; inspect terminal on failure'})+'\n')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    from isaacsim.core.version import get_version
    import carb
    import isaaclab
    import isaaclab.sim as sim_utils
    lab_root = Path(os.environ['ISAACLAB_ROOT']).expanduser().resolve()
    if lab_root not in Path(isaaclab.__file__).resolve().parents:
        raise RuntimeError('Simulator imported Isaac Lab from a different checkout')
    lab_version = (lab_root/'VERSION').read_text().strip()
    if lab_version != args.expected_lab:
        raise RuntimeError(f'Expected Isaac Lab {args.expected_lab}, got {lab_version}')
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.04, device=args.device))
    sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
    sim.reset()
    for _ in range(3):
        sim.step()
    versions = {}
    for name in ('isaaclab', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {'validated': True, 'isaacsim_runtime': list(get_version()),
              'isaaclab_version_file': lab_version, 'isaaclab_source': str(lab_root),
              'python': sys.version, 'executable': sys.executable, 'dependencies': versions,
              'scope': 'Simulator startup and three physics steps; DOM camera/assets verified by the pilot.'}
    tokens = carb.tokens.get_tokens_interface()
    report['kit_paths'] = {name: tokens.resolve('${' + name + '}')
                           for name in ('cache', 'data', 'logs', 'omni_cache', 'omni_logs', 'omni_global_cache')}
    report['asset_root'] = carb.settings.get_settings().get('/persistent/isaac/asset_root/cloud')
    if report['isaacsim_runtime'][0] != args.expected_sim:
        raise RuntimeError(f'Expected Sim {args.expected_sim} for this declared configuration')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
finally:
    app.close()
