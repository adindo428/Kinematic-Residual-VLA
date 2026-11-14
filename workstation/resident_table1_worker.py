"""Resident native policy workers; completion barriers replace blind cooldowns."""
import logging
import os
from pathlib import Path
import queue
import sys
import time

def minimum_retention():
    import json
    request=Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations/krv-full-original-dom-20261106/resident_schedule/request.json'
    value=float(json.loads(request.read_text()).get('minimum_latency_retention',.90)) if request.exists() else .90
    assert 0<value<=1
    return value


def drain(output):
    while True:
        try:output.get_nowait()
        except queue.Empty:return


def session_loop(inbox,output,control,compute,on_activate=lambda path:None,probe=None,on_place=None):
    """Control commands are handled only after the current chunk has completed."""
    active=False
    while True:
        if control.poll(0 if active else .05):
            command=control.recv();kind=command['kind']
            if kind=='close':return
            if kind in ('idle','activate'):
                inbox.clear();drain(output)
                active=kind=='activate'
                if active:on_activate(command.get('log'))
                control.send({'ready':True,'active':active})
                continue
            if kind=='probe':
                assert not active
                control.send(probe(command))
                continue
            if kind=='place':
                assert not active
                control.send(on_place(command['device']))
                continue
            raise ValueError('Unknown resident command')
        if not active:continue
        latest=inbox.get('obs')
        if latest is None:
            time.sleep(.001)
            continue
        inbox.clear()
        result=compute(*latest)
        try:output.put_nowait(result)
        except queue.Full:logging.warning('The output queue is full. Skipping an action.')


def worker(spec,inbox,output,control,log_path):
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,'a',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        sys.path.insert(0,spec['source'])
        import torch
        from safetensors.torch import load_model
        from utils.helpers import get_policy_cfg
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        from capture_table1_runtime import stats
        logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s',force=True)
        torch.set_num_threads(2)
        cfg=get_policy_cfg(cfg_file=Path(spec['checkpoint'])/'config.json');cfg.enable_streaming=False
        policy=DynamicVLAPolicy.from_pretrained(spec['checkpoint'],config=cfg).cuda().eval()
        load_model(policy,str(Path(spec['checkpoint'])/'model.safetensors'),strict=True)
        assert str(Path(sys.modules[DynamicVLAPolicy.__module__].__file__).resolve()).startswith(str(Path(spec['source']).resolve()))
        dummy={k:torch.zeros(1,cfg.n_obs_steps,*f.shape,device='cuda') for k,f in cfg.input_features.items()}
        dummy['task']=['dummy text input'];dummy['observation.environment_state']=torch.zeros(1,cfg.n_obs_steps,9,device='cuda')
        dummy.update(policy.extract_residual_batch_fields(dummy))
        if cfg.residual_phase_gate_mode=='causal_rules':dummy[cfg.residual_phase_gate_key]=torch.ones(1,cfg.chunk_size,1,device='cuda')
        with torch.inference_mode():policy._get_action_chunk(dummy)
        output.put({'initialized':True,'gpu_allocated':torch.cuda.memory_allocated(),'gpu_reserved':torch.cuda.memory_reserved(),
                    'gpu_total':torch.cuda.get_device_properties(0).total_memory,
                    'source':spec['source'],'checkpoint':spec['checkpoint'],'torch_seed':torch.initial_seed()})
        handler=None
        def activate(path):
            nonlocal handler
            logger=logging.getLogger()
            if handler is not None:logger.removeHandler(handler);handler.close()
            handler=logging.FileHandler(path)
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s'))
            logger.addHandler(handler)
        def compute(batch,noise=None):
            with torch.inference_mode():
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                index=int(batch['index']);state=batch['observation.state'][:,-1:,:].clone()
                prepared=policy._prepare_batch(batch)
                actions=policy._get_action_chunk(prepared,None if noise is None else noise.cuda())
                if cfg.use_delta_action:actions[...,:actions.shape[-1]-1]+=state[...,:actions.shape[-1]-1]
                if not torch.isfinite(actions).all():raise ValueError('Nonfinite resident actions')
                return {'actions':actions.transpose(0,1).cpu(),'index':index}
        def probe(command):
            batch={k:torch.zeros(1,cfg.n_obs_steps,*f.shape) for k,f in cfg.input_features.items()}
            batch.update(task=['Place the moving object into the container.'],index=0)
            batch['observation.environment_state']=torch.zeros(1,cfg.n_obs_steps,9)
            batch['observation.environment_state'][...,6:9]=torch.tensor([.1,.02,-.01])
            if cfg.residual_phase_gate_mode=='causal_rules':batch[cfg.residual_phase_gate_key]=torch.ones(1,cfg.chunk_size,1)
            noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,generator=torch.Generator().manual_seed(42))
            actual=compute(batch,noise)['actions'].transpose(0,1)
            gpu={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.no_grad():
                policy.reset();reference=policy.predict_action_chunk(gpu,noise=noise.cuda())
                if cfg.use_delta_action:reference[...,:6]+=gpu['observation.state'][:,-1:,:6]
            torch.testing.assert_close(actual,reference.cpu(),rtol=0,atol=0)
            for _ in range(5):compute(batch,noise)
            measurements=[]
            for _ in range(command.get('samples',20)):
                tick=time.perf_counter();compute(batch,noise);measurements.append(1000*(time.perf_counter()-tick))
            import hashlib
            digest=hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            return {'passed':True,'same_noise_native_actions_exact':True,'reference_grad_mode':'native_no_grad',
                    'action_sha256':digest,'measurements_ms':measurements,**stats(measurements),
                    'gpu_allocated':torch.cuda.memory_allocated(),'gpu_reserved':torch.cuda.memory_reserved()}
        def place(device):
            torch.cuda.synchronize();policy.to(device)
            if device=='cpu':torch.cuda.empty_cache()
            else:torch.cuda.synchronize()
            return {'ready':True,'device':device,'gpu_allocated':torch.cuda.memory_allocated()}
        session_loop(inbox,output,control,compute,activate,probe,place)


class ResidentFront:
    def __init__(self,cfg,inbox,output,control,streaming_method,reset_method):
        from types import MethodType
        self.config=cfg;cfg.enable_streaming=True
        self.q_in=inbox;self.q_out=output;self.control=control;self.log=None
        self.select_action=MethodType(streaming_method,self)
        self._native_reset=MethodType(reset_method,self)
    def command(self,kind,**kw):
        self.control.send({'kind':kind,**kw})
        if not self.control.poll(180):raise TimeoutError('Resident worker control timeout')
        return self.control.recv()
    def idle(self):return self.command('idle')
    def reset(self):
        self.idle();self._native_reset();self.command('activate',log=self.log)


class ResidentPool:
    def __init__(self,specs,directory,profile=False):
        import multiprocessing as mp
        sys.path.insert(0,specs[0]['source'])
        from utils.helpers import get_policy_cfg
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        self.ctx=mp.get_context('spawn');self.manager=self.ctx.Manager();self.fronts={};self.processes=[];self.receipts={}
        self.profile={};self.devices={};self.capacity=len(specs);self.lru=[]
        for index,spec in enumerate(specs):
            # A synchronous manager queue has no delayed feeder publication:
            # after idle is acknowledged, no old chunk can appear later.
            inbox=self.manager.dict();output=self.manager.Queue(maxsize=1);parent,child=self.ctx.Pipe()
            process=self.ctx.Process(target=worker,args=(spec,inbox,output,child,str(Path(directory)/(spec['key']+'.log'))),daemon=True)
            process.start();self.processes.append(process)
            initialized=output.get(timeout=300);assert initialized['initialized']
            cfg=get_policy_cfg(cfg_file=Path(spec['checkpoint'])/'config.json')
            front=ResidentFront(cfg,inbox,output,parent,DynamicVLAPolicy._get_streaming_action,DynamicVLAPolicy.reset)
            self.fronts[spec['key']]=front;self.receipts[spec['key']]=initialized
            self.devices[spec['key']]='cuda';self.lru.append(spec['key'])
            Path(directory).mkdir(parents=True,exist_ok=True)
            import json
            (Path(directory)/'loading.json').write_text(json.dumps({'loaded':list(self.fronts),'target':len(specs),'receipts':self.receipts},indent=2)+'\n')
        if profile:self.qualify()
    def set_capacity(self,count):
        self.idle();self.capacity=count
        for i,key in enumerate(self.fronts):
            device='cuda' if i<count else 'cpu'
            if self.devices[key]!=device:
                self.fronts[key].command('place',device=device);self.devices[key]=device
    def select(self,key):
        self.idle()
        if self.devices[key]!='cuda':
            residents=[k for k in self.lru if self.devices[k]=='cuda']
            while len(residents)>=self.capacity:
                victim=residents.pop(0);self.fronts[victim].command('place',device='cpu');self.devices[victim]='cpu'
            self.fronts[key].command('place',device='cuda');self.devices[key]='cuda'
        self.lru.remove(key);self.lru.append(key)
        return self.fronts[key]
    def qualify(self):
        from capture_table1_runtime import stats
        retention=minimum_retention()
        first=next(iter(self.fronts));front=self.fronts[first];trials=[];reference=None
        if len(self.fronts)==1:
            proof=front.command('probe',samples=100)
            self.profile={'single_resident':proof,'all_resident':proof,'gpu_residency_capacity':1,
                          'minimum_latency_retention':retention,'single_policy_only':True,'paired_trials':[]}
            return
        for count in range(len(self.fronts),1,-1):
            samples={1:[],count:[]};hashes=set();blocks=[]
            # Paired ABBA measurement after all workers have initialized avoids
            # comparing a cold first model to a later, differently loaded host.
            for capacity in [1,count,count,1]:
                self.set_capacity(capacity)
                proof=front.command('probe',samples=50)
                hashes.add(proof['action_sha256']);samples[capacity].extend(proof.pop('measurements_ms'))
                blocks.append({'capacity':capacity,**proof})
            assert len(hashes)==1,'Device placement changed fixed-noise actions'
            a,b=stats(samples[1]),stats(samples[count]);reference=a
            ok=b['p50_ms']<=a['p50_ms']/retention and b['p95_ms']<=a['p95_ms']/retention
            trials.append({'capacity':count,'single_policy':a,'candidate':b,'blocks':blocks,'passed':ok})
            if ok:
                self.set_capacity(count)
                self.profile={'single_resident':a,'all_resident':b,'gpu_residency_capacity':count,
                              'minimum_latency_retention':retention,'paired_trials':trials}
                return
        self.set_capacity(1)
        self.profile={'single_resident':reference,'all_resident':reference,'gpu_residency_capacity':1,
                      'minimum_latency_retention':retention,
                      'conservative_single_policy_fallback':True,'paired_trials':trials}
    def idle(self):
        for front in self.fronts.values():front.idle()
    def close(self):
        for front in self.fronts.values():
            try:front.control.send({'kind':'close'})
            except (EOFError,BrokenPipeError):pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=5)
        self.manager.shutdown()
