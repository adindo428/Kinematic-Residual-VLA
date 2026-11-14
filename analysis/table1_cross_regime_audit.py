"""CPU-only saved-result and training-target audit. Never starts a policy or edits runs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'scratch/dom45'
OUT = WORK/'audits/table1-cross-regime-20261112'
LOW = WORK/'evaluations/low-clutter-full-table1-20261102'
FULL = WORK/'evaluations/krv-full-original-dom-20261106'
DIMS = ['CR','DA','LS','VU','SR','MP','VG','MG','DR']


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write(name, value): (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def paired(a, b):
    aa = {(r['env_name'], r['trial']): r for r in a}
    bb = {(r['env_name'], r['trial']): r for r in b}
    assert len(aa) == len(a) and len(bb) == len(b)
    keys = sorted(aa.keys() & bb.keys())
    counts = Counter(); cases = defaultdict(list); jointly = []
    for key in keys:
        x,y = aa[key],bb[key];sx,sy = bool(x['success']),bool(y['success'])
        counts['both' if sx and sy else 'baseline_only' if sx else 'kr_only' if sy else 'neither'] += 1
        cases[(x['dimension'],key[0])].append(int(sy)-int(sx))
        if sx and sy: jointly.append(y['task_time_s']-x['task_time_s'])
    # Resample complete scenario clusters within each of the nine fixed conditions.
    # This does not measure training-seed uncertainty or policy-noise pairing.
    rng = np.random.default_rng(20261112); bootstrap = np.zeros(10000)
    for dim in DIMS:
        vals = np.array([np.mean(v) for (d,_),v in cases.items() if d == dim])
        assert len(vals) == 10
        weight=sum(len(v) for (d,_),v in cases.items() if d==dim)/len(keys)
        ns=np.array([len(v) for (d,_),v in cases.items() if d==dim])
        draw=rng.integers(0,10,size=(10000,10))
        bootstrap += ((vals[draw]*ns[draw]).sum(axis=1)/ns[draw].sum(axis=1))*weight
    return {'matched_trials':len(keys), 'baseline_percent':100*sum(aa[k]['success'] for k in keys)/len(keys),
            'kr_percent':100*sum(bb[k]['success'] for k in keys)/len(keys),
            'delta_pp':100*sum(bb[k]['success']-aa[k]['success'] for k in keys)/len(keys),
            'scenario_bootstrap_95_pp':(100*np.quantile(bootstrap,[.025,.975])).tolist(),
            'outcomes':dict(counts), 'joint_success_mean_time_delta_s':float(np.mean(jointly)) if jointly else None,
            'by_condition_delta_pp':{d:100*np.mean([np.mean(v) for (dim,_),v in cases.items() if dim==d]) for d in DIMS}}


def result_audit():
    records = {}
    for model in ['dom100_best','dom100_latest','krv100_best','krv100_latest','smolvla_baseline','smolvla_kr','pi05_baseline','pi05_kr']:
        aggregate = LOW/'resident_adapters-20261108/aggregates'/model/'trial_results.jsonl'
        if aggregate.exists(): rr=rows(aggregate)
        else:
            rr=[]
            for p in sorted((LOW/'results'/model/'full').glob('*/trial_results.jsonl')):
                rr += [{**r,'dimension':p.parent.name.split('_',1)[1]} for r in rows(p)]
        assert len(rr)==1800, (model,len(rr))
        records[model]=rr
    live = read(LOW/'vlash-grasp-gating-20261111/live_results.json')
    for k,rr in live['records'].items():
        assert len(rr)==900
        records[k]=rr
    pairs=[('dom100_best','krv100_best'),('dom100_latest','krv100_latest'),('smolvla_baseline','smolvla_kr'),
           ('pi05_baseline','pi05_kr'),('vlash_baseline_fixed','vlash_kr_fixed'),('vlash_kr_fixed','vlash_kr_gated')]
    low_results={a+'__'+b:paired(records[a],records[b]) for a,b in pairs}
    selection=read(LOW/'selection.json')['changes']; changed={t['name'] for t in selection if t['removed_objects']}
    groups={}
    for a,b in pairs:
        groups[a+'__'+b]={}
        for label,scenes in [('capped18',changed),('unchanged72',{t['name'] for t in selection}-changed)]:
            ar=[r for r in records[a] if r['env_name'] in scenes];br=[r for r in records[b] if r['env_name'] in scenes]
            groups[a+'__'+b][label]={'trials_per_model':len(ar),'baseline_percent':100*np.mean([r['success'] for r in ar]),
                                    'kr_percent':100*np.mean([r['success'] for r in br])}
    full={}
    ledgers=[FULL/'resident_schedule/ledger.json',FULL/'resident_schedule/pass02/ledger.json']
    extra=read(FULL/'resident_schedule/additional_models.json')
    ledgers += [Path(m['control_root'])/'ledger.json' for m in extra]
    for path in ledgers:
        ledger=read(path)
        for model, dims in ledger['sources'].items():
            rr=full.setdefault(model,[])
            for dim, sources in dims.items():
                for s in sources:
                    p=Path(s['folder'])/'trial_results.jsonl'
                    if s.get('prefix'): assert hashlib.sha256(p.read_bytes()).hexdigest()==s['records_sha256']
                    pass_offset=10 if path.parent.name=='pass02' else 0
                    rr += [{**r,'trial':r['trial']+s.get('repeat_offset',0)+pass_offset,'dimension':dim} for r in rows(p)]
    first={k:[r for r in rr if r['trial']<=10] for k,rr in full.items()}
    for k,rr in first.items(): assert len(rr)==900,(k,len(rr))
    full_pairs={a+'__'+b:paired(full[a],full[b]) for a,b in [('domv_best','krv_best_constant')]}
    first_pairs={a+'__'+b:paired(first[a],first[b]) for a,b in [
        ('domv_best','krv_best_constant'),('domv_best','krv_latest_constant'),('krv_latest_constant','krv_latest_gated')]}
    full_summary={k:{'first900_success_percent':100*np.mean([r['success'] for r in first[k]]),
                     'first900_by_condition':{d:100*np.mean([r['success'] for r in first[k] if r['dimension']==d]) for d in DIMS},
                     'all_available_trials':len(rr)} for k,rr in full.items()}
    write('paired-results.json',{'low_clutter':low_results,'low_clutter_capped_vs_unchanged':groups,
          'full_scale_all_common':full_pairs,'full_scale_balanced_first900':first_pairs,'full_scale_summary':full_summary,
          'uncertainty':'10,000 paired scenario-cluster bootstrap resamples, stratified over nine conditions. Does not include training-seed variance. Model noise is not paired.'})


def dataset_audit():
    fullfamily=WORK/'training/krv100-full-20261010/family'
    all_episodes={e['episode_index']:e for e in read(fullfamily/'episodes.json')}
    families={'20k_dom':WORK/'training/fresh20k-20261025/family',
              '20k_krv':WORK/'training/krv100-fresh20k-h200-20261029/family',
              'full_krv':fullfamily,'full_domv':WORK/'training/dom-full-resume-deltaai-20261110/local-cpu-inputs'}
    sets={}; result={}
    for name,family in families.items():
        splits=read(family/'split.json')['splits'];sets[name]={k:set(v) for k,v in splits.items()}
        manifest=read(family/'manifest.json');rr=[all_episodes[i] for i in splits['train']]
        tasks=Counter(r['task'] for r in rr);frames=Counter()
        for r in rr: frames[r['task']]+=r['length']
        groups={k:{all_episodes[i]['duplicate_id'] for i in ids} for k,ids in splits.items()}
        result[name]={'counts':{k:len(v) for k,v in splits.items()},'train_frames':sum(frames.values()),
                      'task_episodes':dict(tasks),'task_frames':dict(frames),
                      'episode_mean_speed_quantiles_mps':np.quantile([r['mean_speed'] for r in rr],[0,.25,.5,.75,.95,1]).tolist(),
                      'velocity_mean':manifest['velocity_mean'],'velocity_std':manifest['velocity_std'],
                      'duplicate_groups_train_validation_overlap':len(groups['train']&groups['validation']),
                      'duplicate_groups_train_test_overlap':len(groups['train']&groups['test'])}
    result['overlap']={}
    for a,b in [('20k_dom','20k_krv'),('20k_dom','full_krv'),('20k_dom','full_domv'),('full_domv','full_krv')]:
        result['overlap'][a+'__'+b]={k:{'shared':len(sets[a][k]&sets[b][k]),'a_only':len(sets[a][k]-sets[b][k]),
                                    'b_only':len(sets[b][k]-sets[a][k])} for k in ['train','validation','test']}
    write('dataset-comparison.json',result)
    return families,sets,all_episodes


def target_audit(families,sets,episodes):
    rng=np.random.default_rng(20261112)
    sources={name:(WORK/'cache/huggingface/lerobot/hzxie/DOM',rng.choice(sorted(sets[name]['train']),512,replace=False).tolist(),
                   read(families[name]/'manifest.json')['velocity_mean']) for name in ['20k_dom','full_krv','full_domv']}
    sources['tracking']=(OUT/'tracking-dataset',read(ROOT/'experiments/tracking75/source/configs/dom_franka_tracking_split.json')['train'],
                         [-.0002714536565211347,-8.079752708067645e-06,.000053059701140341534])
    result={}
    for name,(data,ids,mean) in sources.items():
        aa=[];dd=[];vv=[];phases=[];task=[];forecast=[];lead=[];moments=[]
        for episode in ids:
            p=data/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
            t=pq.read_table(p,columns=['action','observation.state','observation.environment_state'],use_threads=False).to_pydict()
            a=np.array(t['action']);s=np.array(t['observation.state']);e=np.array(t['observation.environment_state'])
            indices=np.arange(2,len(a)-19)
            if not len(indices):continue
            future=indices[:,None]+np.arange(20)[None,:]
            raw=a[future,:3]-s[indices,None,:3]
            drift=(e[indices,6:9]-np.array(mean))[:,None,:]*(np.arange(20)[None,:,None]*.04)
            aa.append(raw);dd.append(drift);vv.append(e[indices,6:9])
            residual=raw-drift
            moments.append((len(raw),raw.sum(axis=0).ravel(),(raw*raw).sum(axis=0).ravel(),
                            residual.sum(axis=0).ravel(),(residual*residual).sum(axis=0).ravel()))
            forecast.append(e[indices+19,:3]-e[indices,:3]-.76*e[indices,6:9])
            # Verify the published +5-frame action-label relationship on retained interior rows.
            lead.append(np.linalg.norm(a[:-5,:3]-s[5:,:3],axis=1))
            phase_path=WORK/f'training/krv100-full-20261010/family/phase_cache/{episode:06d}.npz'
            phases.append(np.load(phase_path)['phase'][indices] if name!='tracking' else np.zeros(len(indices),int))
            task += [episodes[episode]['task'] if name!='tracking' else 'tracking']*len(indices)
        a=np.concatenate(aa);d=np.concatenate(dd);v=np.concatenate(vv);phase=np.concatenate(phases);task=np.array(task)
        def stats(mask):
            x=a[mask];y=d[mask];r=x-y
            av=float(x.var(axis=0).sum());rv=float(r.var(axis=0).sum());dv=float(y.var(axis=0).sum())
            cov=float(((x-x.mean(axis=0))*(y-y.mean(axis=0))).mean(axis=0).sum())
            return {'anchors':len(x),'raw_xyz_variance_sum_m2':av,'residual_xyz_variance_sum_m2':rv,
                    'residual_to_raw_variance_ratio':rv/av,'drift_variance_sum_m2':dv,'raw_drift_covariance_sum_m2':cov,
                    'raw_xyz_rms_m':float(np.sqrt(np.mean(x*x))), 'residual_xyz_rms_m':float(np.sqrt(np.mean(r*r))),
                    'final_drift_norm_m_quantiles':np.quantile(np.linalg.norm(y[:,-1],axis=1),[.5,.9,.99]).tolist()}
        # Episode-cluster bootstrap quantifies uncertainty from the sampled trajectories.
        weights=rng.multinomial(len(moments),np.full(len(moments),1/len(moments)),size=2000)
        count=weights@np.array([m[0] for m in moments])
        def boot_var(i):
            sums=weights@np.stack([m[i] for m in moments])
            squares=weights@np.stack([m[i+1] for m in moments])
            return (squares/count[:,None]-(sums/count[:,None])**2).sum(axis=1)
        ratio_ci=np.quantile(boot_var(3)/boot_var(1),[.025,.975]).tolist()
        horizon_ratios={str(j):float((a[:,j]-d[:,j]).var(axis=0).sum()/a[:,j].var(axis=0).sum()) for j in [0,4,9,19]}
        result[name]={'sampled_episode_ids':ids,'sampling':'All nonpadded 20-action anchors at indices >=2 in a fixed uniform episode sample; no RGB decode or model inference.',
                      'episode_bootstrap_variance_ratio_95':ratio_ci,'by_horizon_index_variance_ratio':horizon_ratios,
                      'all':stats(np.ones(len(a),bool)), 'by_task':{k:stats(task==k) for k in sorted(set(task))},
                      'by_causal_phase':{str(k):stats(phase==k) for k in sorted(set(phase)) if sum(phase==k)>10},
                      'speed_mps_quantiles':np.quantile(np.linalg.norm(v,axis=1),[.5,.9,.99]).tolist(),
                      'constant_velocity_0p76s_object_forecast_error_m_quantiles':np.quantile(np.linalg.norm(np.concatenate(forecast),axis=1),[.5,.9,.99]).tolist(),
                      'action_vs_state_plus5_xyz_error_m_quantiles':np.quantile(np.concatenate(lead),[.5,.9,.99]).tolist()}
        print(name,'target variance ratio',result[name]['all']['residual_to_raw_variance_ratio'],flush=True)
    write('target-statistics.json',result)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    result_audit()
    families,sets,episodes=dataset_audit()
    target_audit(families,sets,episodes)
