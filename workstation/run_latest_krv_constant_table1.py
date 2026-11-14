#!/usr/bin/env python3
"""Queue the ungated latest KRV checkpoint without interrupting the active cohort."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import time

import resident_table1_schedule as first
import resident_table1_second_pass as second
import run_resident_table1_queue as queue

low=first.low
PARENT=first.ROOT
EVALUATION=first.krv.ROOT/'variants/krv100_after_epoch004_partial005_constant'
CONTROL=EVALUATION/'resident_schedule'
MODEL_KEY='krv_latest_constant'
FROZEN_PROMPTS={}


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    if (CONTROL/'preparation_complete.json').exists():return low.read(CONTROL/'model.json')
    gated=low.read(PARENT/'ledger.json')['models'][-1]
    checkpoint=Path(gated['parent_checkpoint'])
    config=low.read(checkpoint/'config.json');other=low.read(Path(gated['checkpoint'])/'config.json')
    assert config['residual_phase_gate_mode']=='constant' and config['enable_residual_flow']
    assert config['condition_on_object_velocity'] and config.get('residual_target_lead_s',0)==0
    assert {k for k in set(config)|set(other) if config.get(k)!=other.get(k)}=={'residual_phase_gate_mode'}
    assert low.sha(checkpoint/'model.safetensors')==gated['weight_sha256']
    model={**gated,'key':MODEL_KEY,'label':'KRV100_Full latest (4 epochs + partial epoch 5, constant gate)',
           'root':str(EVALUATION),'checkpoint':str(checkpoint),'gated':False,
           'config_sha256':low.sha(checkpoint/'config.json')}
    EVALUATION.mkdir(parents=True,exist_ok=True)
    protocol=deepcopy(low.read(Path(gated['root'])/'protocol.json'))
    protocol.update(checkpoint=str(checkpoint),trained_checkpoint=str(checkpoint),
                    checkpoint_config_sha256=model['config_sha256'],method_label=model['label'],
                    intervention={'neural_velocity':True,'residual_flow':True,'phase_gate_mode':'constant',
                                  'target_lead_s':0,'paired_gated_weights_sha256':model['weight_sha256']})
    low.write(EVALUATION/'protocol.json',protocol)
    for name in ['test_lists','two_pass_plan.json']:
        source=Path(gated['root'])/name
        if source.is_dir():
            target=EVALUATION/name;target.mkdir(exist_ok=True)
            for path in source.glob('*.txt'):(target/path.name).write_bytes(path.read_bytes())
        else:(EVALUATION/name).write_bytes(source.read_bytes())
    low.write(CONTROL/'model.json',model)
    low.write(CONTROL/'ledger.json',{'models':[model],'sources':{MODEL_KEY:{d:[] for d in low.DIMS}},
              'accepted_jobs':[],'budget':{'per_column':100,'per_checkpoint':900}})
    request=deepcopy(low.read(PARENT/'request.json'))
    names=['resident_table1_worker.py','resident_table1_server.py','resident_table1_client.py',
           'resident_table1_schedule.py','resident_table1_second_pass.py','run_resident_table1_queue.py',
           'run_latest_krv_constant_table1.py','pilot_resident_table1.py','table1_resume_records.py','table1_two_pass.py']
    request.pop('second_pass_root',None)
    request.update(models=[model],remaining_ten_repeats_deferred=True,user_authorized=True,
                   second_pass_authorized=False,run_passes=[1],
                   schedule='After the current three-checkpoint pass: ten repeats per scene in pass one; repeats 11–20 deferred.',
                   code_sha256={str(low.REPO/'workstation'/n):low.sha(low.REPO/'workstation'/n) for n in names})
    low.write(CONTROL/'request.json',request)
    low.write(CONTROL/'status.json',{'stage':'queued_after_current_pass','updated_utc':stamp(),
                                   'model':MODEL_KEY,'total_trials':900,'remaining_trials_deferred':900})
    (EVALUATION/'table1_comparison.md').write_text(
        '# KRV100_Full latest — constant gate\n\n'
        '| Method | Status | Trials |\n|---|---|---:|\n'
        f"| {model['label']} | Queued after the current three-checkpoint pass | 0/900 |\n\n"
        'Same weights as the latest gated row; KR remains active throughout. '
        'Ten repeats across each of the 90 original configurations now; the remaining ten repeats are deferred.\n')
    registry=PARENT/'additional_models.json'
    entries=low.read(registry) if registry.exists() else []
    assert not any(m['key']==MODEL_KEY for m in entries)
    low.write(registry,[*entries,{**model,'control_root':str(CONTROL)}])
    low.write(CONTROL/'preparation_complete.json',{'ready':True,'updated_utc':stamp()})
    return model


def source_prompts():
    # First-pass prompts are complete even if the older cohort is now paused.
    prior=low.read(PARENT/'ledger.json');tail=low.read(PARENT/'pass02/ledger.json')
    result=first.saved_prompts(prior)
    for scene,values in first.saved_prompts(tail).items():
        result.setdefault(scene,{}).update({str(int(k)+10):v for k,v in values.items()})
    assert len(result)==90 and all({str(i) for i in range(1,11)}<=set(v) for v in result.values())
    return result


def execute_phase(phase):
    first.ROOT=CONTROL
    second.ROOT=CONTROL/'pass02'
    schedule=first if phase==1 else second
    original_make_plan=schedule.make_plan
    def make_plan(ledger,directory):
        plan=original_make_plan(ledger,directory)
        expected={t['name']:t['config_sha256'] for t in low.read(EVALUATION/'protocol.json')['tests']}
        for job in plan['jobs']:assert job['scene_sha256']==expected[Path(job['scene']).stem]
        plan['saved_prompts']=FROZEN_PROMPTS
        plan['second_ten_repeats_deferred']=low.read(CONTROL/'request.json').get('remaining_ten_repeats_deferred',True)
        low.write(Path(directory)/'plan.json',plan)
        return plan
    def publish(stage,**kw):
        if stage=='first_pass_queue_complete':stage='first_pass_complete' if phase==1 else 'complete'
        kw.pop('remaining_repeats_deferred',None)
        value={'stage':stage,'pass_index':phase,'updated_utc':stamp(),**kw}
        low.write(CONTROL/'status.json',value)
        if phase==2:low.write(second.ROOT/'status.json',value)
        from render_resident_table1 import render
        render()
    def prepare_phase():
        ledger=schedule.freeze()
        request_path=schedule.ROOT/'request.json'
        if not request_path.exists():low.write(request_path,low.read(CONTROL/'request.json'))
        return ledger
    schedule.make_plan=make_plan
    queue.schedule=schedule;queue.ROOT=schedule.ROOT;queue.prepare=prepare_phase;queue.publish=publish
    queue.STOP.clear()
    try:queue.main()
    finally:schedule.make_plan=original_make_plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    model=prepare()
    if args.prepare_only:return
    # queue.main parses argv too; its only recognized optional flag is not used.
    sys.argv=sys.argv[:1]
    signal.signal(signal.SIGTERM,lambda *_:queue.STOP.set())
    signal.signal(signal.SIGINT,lambda *_:queue.STOP.set())
    while True:
        state=low.read(PARENT/'status.json')
        immediate=low.read(CONTROL/'request.json').get('start_immediately',False)
        if (state.get('stage')=='complete' and state.get('pass_index')==2) or (immediate and state.get('stage')=='paused'):
            # Completion is published just before the old owner releases its
            # lock. Wait for that boundary before launching another simulator.
            with (low.ROOT/'queue.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:break
        if state.get('stage') in ('paused','failed'):
            low.write(CONTROL/'status.json',{'stage':'waiting_for_current_pass_recovery','updated_utc':stamp()})
        if queue.STOP.wait(15):return
    if not immediate:
        for peer in low.read(PARENT/'ledger.json')['models']:
            summary=low.read(Path(peer['root'])/'summary.json')
            assert summary['complete'] and summary['verified_trials']==1800
    else:
        second.validate_first_pass(low.read(PARENT/'ledger.json'))
    checkpoint=Path(model['checkpoint'])
    assert low.sha(checkpoint/'model.safetensors')==model['weight_sha256']
    assert low.sha(checkpoint/'config.json')==model['config_sha256']
    global FROZEN_PROMPTS
    FROZEN_PROMPTS=source_prompts();low.write(CONTROL/'matched_prompts.json',FROZEN_PROMPTS)
    # Reload only the passive renderer; the active cohort has already ended.
    subprocess.run(['systemctl','--user','restart','krvla-table1-displays-20261106.service'],check=True)
    phases=low.read(CONTROL/'request.json').get('run_passes',[1])
    assert phases in ([1],[1,2])
    for phase in phases:execute_phase(phase)
    if phases==[1]:
        summary=low.read(EVALUATION/'first_pass_summary.json')
        assert summary['first_pass_complete'] and summary['verified_trials']==900
        low.write(CONTROL/'status.json',{'stage':'first_pass_complete_deferred','pass_index':1,
                  'updated_utc':stamp(),'verified_trials':900,'remaining_trials_deferred':900})
        from render_resident_table1 import render
        render()
    else:
        summary=low.read(EVALUATION/'summary.json')
        assert summary['complete'] and summary['verified_trials']==1800


if __name__=='__main__':main()
