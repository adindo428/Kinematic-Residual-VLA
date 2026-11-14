#!/usr/bin/env python3
"""Run the explicitly requested 3-case CR gate intervention, separate from Table 1."""
import json,os,signal,statistics,subprocess,threading,time
from datetime import datetime,timezone
from pathlib import Path
import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
from monitor_table1_latency import log_samples
from capture_table1_runtime import stats
ROOT=Path(os.environ.get('DOM_GATE_DIAGNOSTIC_ROOT',str(low.WORK/'evaluations/krv100-full-causal-gate-cr-20261106')))
STOP=threading.Event()

def read_rows(path):
    if not path.exists():return []
    # Allow live reads while the other process is appending a record.
    lines=path.read_text().splitlines();rows=[]
    for line in lines:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
    return rows

def status(state,**extra):
    low.write(ROOT/'status.json',{'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),'model':low.read(ROOT/'protocol.json')['method_label'],**extra})

def report(state):
    from render_causal_gate_comparison import render
    render()


def main():
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    c=low.read(low.ROOT/'catalog.json');low.validate(c);krv.CATALOG=c
    assert low.read(ROOT/'policy_probe.json')['same_noise_gate_one_matches_constant_exactly']
    cfg=low.read(ROOT/'checkpoint/config.json');assert cfg['residual_phase_gate_mode']=='causal_rules'
    assert low.sha(ROOT/'checkpoint/model.safetensors')==c['models']['krv100_full']['checkpoint_sha256']
    assert low.sha(ROOT/'checkpoint/model.safetensors')==low.read(ROOT/'protocol.json')['checkpoint_weights_sha256']
    protocol=low.read(ROOT/'protocol.json')
    lead_s=float(protocol.get('target_lead_s',0))
    if lead_s:
        assert low.read(ROOT/'timing_probe.json')['passed']
        assert low.read(ROOT/'timing_probe.json')['target_lead_s']==lead_s
        for filename,digest in protocol['diagnostic_code_sha256'].items():
            assert low.sha(low.REPO/filename)==digest,filename
    for t in low.read(ROOT/'protocol.json')['tests']:assert low.sha(t['path'])==t['config_sha256']
    folder=ROOT/'full';assert not folder.exists(),'Existing diagnostic records must not be overwritten';folder.mkdir()
    server,client=krv.commands('original_dom',folder,ROOT/'cases.txt',10,0)
    server[server.index('-p')+1]=str(low.REPO/'workstation/causal_gate_diagnostic_server.py')
    client=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation/causal_gate_diagnostic_client.py'),'--source',c['models']['krv100_full']['source'],'-p',str(ROOT/'checkpoint'),'-a','KRV100_Full_causal_gate_CR','-r','euler','-d','-s','--img_port','27600','--act_port','27601','-o',str(folder/'actions')]
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=c['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),DOM_GATE_TRACE=str(folder/'gate_trace.jsonl'),DOM_GATE_WORKER_TRACE=str(folder/'gate_worker_trace.jsonl'),DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',DOM_REPEAT_OFFSET='0')
    env.pop('DOM_VIDEO_SELECTION_HISTORY',None)
    env['DOM_RESIDUAL_TARGET_LEAD_S']=str(lead_s)
    low.write(folder/'commands.json',{'server':server,'client':client})
    low.write(folder/'runtime.json',{'execution_mode':'serial_diagnostic','simulation_gpu':0,'inference_gpu':1,'repeats':10,'scored_trials':30,'full_suite_inclusion':False,'target_lead_s':lead_s})
    processes=[];changed=time.monotonic();last=-1
    try:
        with (folder/'server.log').open('w') as s,(folder/'client.log').open('w') as cl:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=s,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=cl,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.is_set():raise InterruptedError('Diagnostic paused by user')
                if processes[0].poll() is not None:raise RuntimeError('Diagnostic simulator exited')
                n=len(read_rows(folder/'trial_results.jsonl'))
                if n!=last:last=n;changed=time.monotonic()
                status('evaluating',completed_trials=n,target_trials=30);report('running')
                if time.monotonic()-changed>1200:raise TimeoutError('No trial progress for 20 minutes')
                STOP.wait(5)
            if processes[1].returncode:raise RuntimeError('Diagnostic client failed')
        v=low.runtime.verify_trial_folder(folder,ROOT,ROOT/'cases.txt',10)
        assert read_rows(folder/'gate_worker_trace.jsonl'),'Missing inference-worker gate evidence'
        if lead_s:
            assert all(r['target_lead_s']==lead_s for r in read_rows(folder/'gate_worker_trace.jsonl'))
        low.write(ROOT/'summary.json',{'complete':True,**v,'policy_latency':stats([r['ms'] for r in log_samples(folder/'client.log')]),'inference_only_intervention':True})
        status('complete',completed_trials=30,target_trials=30);report('complete')
    except BaseException as e:status('failed',error=repr(e));report('failed');raise
    finally:
        for p in reversed(processes):low.stop_group(p)

if __name__=='__main__':main()
