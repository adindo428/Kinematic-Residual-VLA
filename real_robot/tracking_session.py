"""YAML launcher. Perception is camera-only; arm defaults to offline plan checking."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .tracking_config import load_config

PROJECT=Path(__file__).resolve().parent.parent

def resolve(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else PROJECT/p

def output_path(section,kind):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    parent=resolve(section['output_root']); parent.mkdir(parents=True,exist_ok=True)
    return parent/(kind+'-'+stamp)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('perception','follow','arm'))
    parser.add_argument('--config',default=str(PROJECT/'real_robot/config.yaml'))
    parser.add_argument('--plan',help='Override scene-reviewed plan path')
    parser.add_argument('--observation',help='Override live observation path for continuous following')
    parser.add_argument('--execute',action='store_true',help='Enable continuous-follow motor goals')
    parser.add_argument('--record-rgbd', action='store_true',
                        help='Perception: save every processed front/side RGB-D pair for grasp recording')
    parser.add_argument('--front-seed-box',type=int,nargs=4,metavar=('X','Y','W','H'))
    parser.add_argument('--side-seed-box',type=int,nargs=4,metavar=('X','Y','W','H'))
    starts=parser.add_mutually_exclusive_group()
    starts.add_argument('--execute-held-start',action='store_true')
    starts.add_argument('--execute-supported-start',action='store_true')
    args=parser.parse_args(); config=load_config(args.config)
    if args.mode=='perception':
        if args.execute_held_start or args.execute_supported_start or args.plan or args.execute or args.observation:
            parser.error('Arm flags cannot be used with camera-only perception')
        section=config['perception']; allowed={'rig','seed_session','output_root','seconds','max_skew_s','save_every','preview','observation_max_age_s','front_depth_range_m','side_depth_range_m'}
        if set(section)-allowed: raise ValueError('Unknown perception setting')
        output=output_path(section,'perception')
        argv=['--rig',str(resolve(section['rig'])),'--output',str(output),
              '--mask-tracker','efficienttam',
              '--seconds',str(section['seconds']),'--max-skew',str(section['max_skew_s']),
              '--save-every',str(section['save_every']),'--observation-max-age',str(section.get('observation_max_age_s',.20))]
        if args.record_rgbd:
            argv[argv.index('--save-every')+1] = '1'
        for role in ('front', 'side'):
            if role+'_depth_range_m' in section:
                argv += ['--'+role+'-depth-range', *map(str, section[role+'_depth_range_m'])]
        if args.front_seed_box is not None or args.side_seed_box is not None:
            if args.front_seed_box is None or args.side_seed_box is None:parser.error('Provide both seed boxes')
            argv+=['--front-seed-box',*map(str,args.front_seed_box),'--side-seed-box',*map(str,args.side_seed_box)]
        elif section.get('seed_session'):argv+=['--seed-session',str(resolve(section['seed_session']))]
        if section.get('preview',True): argv+=['--preview']
        print('Perception output:',output,flush=True)
        print('Live observation feed:',output/'observation.json',flush=True)
        pointer=resolve(section['output_root'])/'latest-perception.json'
        pointer.with_suffix('.tmp').write_text(json.dumps({'observation_file':str(output/'observation.json'),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
        pointer.with_suffix('.tmp').replace(pointer)
        from .multiview_track import main as run
    elif args.mode=='follow':
        if args.record_rgbd: parser.error('--record-rgbd belongs to perception mode')
        if args.front_seed_box is not None or args.side_seed_box is not None:parser.error('Seed boxes belong to perception mode')
        if args.plan or args.execute_held_start or args.execute_supported_start:parser.error('Continuous follow does not use approach-plan flags')
        if args.observation:observation=resolve(args.observation)
        else:
            pointer=resolve(config['perception']['output_root'])/'latest-perception.json'
            if not pointer.exists():parser.error('Start perception first, or provide --observation')
            data=json.loads(pointer.read_text())
            if data['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():parser.error('Perception pointer is from an earlier boot')
            observation=Path(data['observation_file'])
        output=output_path(config['arm'],'follow')
        argv=['--config',str(Path(args.config).resolve()),'--observation',str(observation),'--output',str(output)]
        if args.execute:argv+=['--execute']
        print('Continuous tracker output:',output,flush=True)
        from .aloha_object_servo import main as run
    else:
        if args.record_rgbd: parser.error('--record-rgbd belongs to perception mode')
        if args.execute or args.observation:parser.error('Use follow mode for --execute / --observation')
        section=config['arm']
        if set(section)-{'plan','output_root','seconds'}: raise ValueError('Unknown arm setting')
        if not args.plan and 'plan' not in section:parser.error('Legacy arm mode needs --plan; use follow for continuous tracking')
        argv=['--plan',str(resolve(args.plan or section['plan'])),'--output',str(output_path(section,'arm')),
              '--seconds',str(section['seconds']),'--config',str(Path(args.config).resolve())]
        print('Arm output:',argv[3],flush=True)
        if args.execute_held_start: argv+=['--execute-held-start']
        elif args.execute_supported_start: argv+=['--execute-supported-start']
        else: print('Offline plan validation only; no motor access.',flush=True)
        from .aloha_dynamic_follow import main as run
    try: result=run(argv)
    except ValueError as exc: parser.exit(2,'Cannot start: '+str(exc)+'\n')
    if result: raise SystemExit(result)

if __name__=='__main__': main()
