#!/usr/bin/env python3
"""Make matched, fresh trials and 90 fixed evaluation shards."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tracking_online_common import sha256, write_json
from tracking_checkpoint import checkpoint_files
from tracking_variants import validate_variant, VARIANT_LABELS
from tracking_benchmark import DIMENSIONS, MODELS, BenchmarkPath, SCORING_VERSION, SUCCESS_DEFINITION, TIME_DEFINITION


def prepare(previous, output, objects=None):
    original=json.loads(Path(previous).read_text())
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    models={k:original['models'][k] for k in MODELS}
    metadata={}
    for k, root in models.items():
        root,c,w,p=checkpoint_files(root,require_complete=True)
        validate_variant(SimpleNamespace(**p),k)
        assert bool(p.get('condition_on_object_velocity',False))==(k in ('velocity','coordinate_velocity'))
        metadata[k]={'weights_sha256':sha256(w),'config_sha256':sha256(c)}
        assert metadata[k]['weights_sha256']==original['checkpoint_metadata'][k]['weights_sha256']
    assert sha256(original['layout']['scene_usd'])==original['layout']['scene_usd_sha256']
    assert sha256(original['layout']['object_usd'])==original['layout']['object_usd_sha256']
    objects=Path(objects or '/scratch/aparame/Dynamic_VLA/dom_benchmark/objects')
    vg_objects=[str(objects/'apple'/f'apple{i:02d}.usd') for i in (2,3,4,5)]
    vg_objects+=[str(objects/'unseen/apple99.usd')]
    assets={p:sha256(p) for p in vg_objects}
    protocol={'name':'DynamicVLA Table 1 tracking adaptation v1',
        'paper':'https://arxiv.org/abs/2601.22153',
        'dimensions':DIMENSIONS, 'models':models,'model_labels':{k:VARIANT_LABELS[k] for k in MODELS},
        'checkpoint_metadata':metadata, 'dataset_revision':original['dataset_revision'],
        'setups_per_dimension':10,'trials_per_setup':20,'expected_rollouts_per_model':1800,
        'expected_rollouts':7200,'dimension_order':list(DIMENSIONS),
        'scoring_version':SCORING_VERSION,
        'success_tolerance_m':.03, 'success_aggregation':'arithmetic mean of per-episode time percentages',
        'initial_second_counts_for_success':True, 'binary_success_threshold':None,
        'success_definition':SUCCESS_DEFINITION,
        'time_definition':TIME_DEFINITION,
        'duration_seconds':{'default':10,'LS':30},
        'scope':'Ten seeded workspace setups in ONE training room, not ten distinct DOM scenes. VG tests five held-out apple geometries, not held-out scene generalization. LS is continuous long-horizon tracking, not manipulation sequencing. VU/MP/DR are declared tracking proxies.',
        'inputs':'Two RGB views, fixed tracking instruction, measured robot EE pose; current simulator object velocity for velocity-conditioned models and analytic residual drift. No target pose/future trajectory sent to the neural policy.',
        'timing':'25 Hz RGB, 200 Hz physics/scoring; fixed 0.8 s replanning, world paused during inference. Same adapter, controller and expert warmup as previous comparison.',
        'recording':'All trials retain telemetry/action chunks/initial states; three-view videos for trial 0 of each setup (360 recorded trials). Pilot videos for every pilot trial.',
        'heldout_object_sha256':assets,'pilot_shards':list(range(0,90,10))+[61,62,63,64]}
    write_json(output/'protocol.json',protocol)
    for di,dim in enumerate(DIMENSIONS):
        for env in range(10):
            m={k:copy.deepcopy(original[k]) for k in ('config','layout','train_max_speed_mps','collector_commit','dataset_revision')}
            m.update(models=models,model_labels=protocol['model_labels'],checkpoint_metadata=metadata,
                     benchmark_protocol=protocol['name'],dimension=dim,environment=env,
                     seconds=30 if dim=='LS' else 10,speed_multipliers=[1.],episodes=[])
            if dim=='VG':
                m['layout']['object_usd']=vg_objects[env%5]
                m['layout']['object_usd_sha256']=assets[vg_objects[env%5]]
            # All spatial tests remain in the original conservative reachable box.
            env_rng=np.random.default_rng(np.random.SeedSequence([20261008,env]))
            half=np.array([.025,.045,.025]) if dim=='SR' else np.array([.015,.025,.01])
            center=np.array([.45,0,.16])+env_rng.uniform(-1,1,3)*half
            for trial in range(20):
                seed=int(np.random.SeedSequence([20261008,di,env,trial]).generate_state(1)[0])
                rng=np.random.default_rng(seed)
                amplitude=np.array([.04,.08,.025])*rng.uniform(.55,.95,3)
                omega=2*np.pi*rng.uniform(.1,.28,3)
                phase=rng.uniform(-np.pi,np.pi,3)
                if dim=='CR':
                    amplitude[[0,2]]=0; omega[1]=2*np.pi/4
                amplitude*=min(1,.15/max(np.linalg.norm(amplitude*omega),1e-12),
                               .3/max(np.linalg.norm(amplitude*omega**2),1e-12))
                condition={'dimension':dim,'speed_multiplier':float(rng.uniform(1.5,2)) if dim=='MG' else 1.,
                    'push_direction':[0.,1. if trial%2==0 else -1.,0.],
                    'exposure':float(rng.uniform(.65,1.35)), 'contrast':float(rng.uniform(.75,1.25)),
                    'occluder_rgb':rng.integers(20,220,3).tolist()}
                path=dict(center=center.tolist(),amplitude=amplitude.tolist(),omega=omega.tolist(),phase=phase.tolist())
                bp=BenchmarkPath(path,condition)
                p,v,a=bp.sample(np.arange(round(m['seconds']*200)+1)/200)
                assert np.all(p>np.array([.35,-.15,.085])) and np.all(p<np.array([.55,.15,.235]))
                m['episodes'].append({'episode':trial,'seed':seed,'trajectory':path,'condition':condition,
                    'object_mean_speed_mps':float(np.linalg.norm(v,axis=1).mean()),
                    'object_max_speed_mps':float(np.linalg.norm(v,axis=1).max())})
            write_json(output/'shards'/f'{di*10+env:03d}.json',m)
    print('Prepared 90 matched shards, 1800 trials per model, 7200 total.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--previous',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.previous,a.output)
