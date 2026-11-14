"""Evaluate missing C (velocity-conditioned DOMV + innovation), reusing D/F."""
from pathlib import Path

_parent=Path(__file__).with_name('run_krv_innovation_only.py')
_text=_parent.read_text()
for old,new in {
    'closed-loop-krv-innovation-only-20261112':'closed-loop-domv-innovation-cdf-20261112',
    "'krv_best_constant'":"'domv_best'",
    'from report_krv_innovation_only import render as report':'from report_cdf_ablation import render as report',
    '    report()':"    report(shared=(ROOT/'execution').exists())",
    "if __name__ == '__main__':main()":'',
}.items():
    assert old in _text,old
    _text=_text.replace(old,new)
exec(compile(_text,str(_parent),'exec'))
_prepare=prepare


def prepare():
    existing=(ROOT/'plan.json').exists()
    plan=_prepare()
    if existing:return plan
    historical=low.read(ROOT.parent/'closed-loop-pilot-20261112/plan.json')
    model=deepcopy(next(m for m in historical['models'] if m['key']=='domv_best'))
    cfg=low.read(Path(model['checkpoint'])/'config.json')
    assert cfg['condition_on_object_velocity'] and not cfg['enable_residual_flow']
    assert low.sha(Path(model['checkpoint'])/'model.safetensors')==model['weight_sha256']
    assert low.sha(Path(model['checkpoint'])/'config.json')==model['config_sha256']
    plan['models']=[model]
    for job in plan['jobs']:
        job['model']='domv_best';job['ablation_arm']='C'
        assert job['correction_mode']=='innovation'
    plan['protocol'].update(comparison='C: velocity-conditioned raw-action DOMV + innovation; reuse completed D and F',
        checkpoint=model['checkpoint'],checkpoint_sha256=model['weight_sha256'],
        native_kr_mode='disabled in DOMV; constant retained in existing KRV D/F',
        observations='Unchanged native DOMV neural velocity inputs and oracle velocity for identical innovation controller',
        fully_matched_training=False,training_split_episodes={'DOMV':197198,'KRV':203151},
        attribution_limit='Different training split, normalization and source histories; checkpoint-level comparison only',
        user_steering='Test whether F beats C and D; no redundant baseline reruns')
    for p in (Path(__file__),_parent,Path(__file__).with_name('report_cdf_ablation.py')):
        plan['files_sha256'][str(p.resolve())]=low.sha(p)
    low.write(ROOT/'plan.json',plan)
    low.write(ROOT/'protocol.json',{**plan['protocol'],'tests':plan['tests']})
    low.write(ROOT/'checkpoint_audit.json',dict(C=dict(key=model['key'],checkpoint=model['checkpoint'],
        condition_on_object_velocity=True,enable_residual_flow=False,config_sha256=model['config_sha256']),
        D_and_F=dict(key='krv_best_constant',condition_on_object_velocity=True,enable_residual_flow=True,
        phase_gate='constant'),fully_matched_training=False,training_split_episodes={'DOMV':197198,'KRV':203151},
        velocity_normalization_matches=False))
    (ROOT/'README.md').write_text('# C / D / F comparison\n\n'
        'Run only missing C: DOMV full-data epoch-3 validation-best, neural velocity conditioning enabled, KR disabled, '
        'plus exactly the same velocity-innovation controller as F. 75 cases at 0.10/0.50/1.00 m/s. '
        'Reuse historical D and completed F. Wait for the shared GPU lock; do not interrupt the existing no-restoration ablation.\n\n'
        'This is not a fully matched training experiment: DOMV used 197,198 episodes, KRV 203,151, with different saved velocity statistics. '
        'The released pretrained DOM is not the velocity-conditioned control. '
        'Existing D and F already show 20/75 versus 17/75; the quoted improvement is hypothetical.\n\n'
        '[Live comparison and matched-case statistics](comparison.md). No training or baseline reruns are launched.\n')
    status('prepared',completed=0)
    return plan


if __name__=='__main__':main()
