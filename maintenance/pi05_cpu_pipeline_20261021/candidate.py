"""Benchmark-only CPU prompt preparation and copy-stream prefetch candidates.

No production module is edited. Exact tokens are checked against the frozen
GPU-state implementation on actual H200 before measuring either candidate.
"""
import json
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from policy import Policy
from common import drift,coordinates


def state_prompt(tasks,normalized_state):
    padded=torch.nn.functional.pad(normalized_state,(0,32-normalized_state.shape[-1]))
    bins=torch.linspace(-1,1,257,device=normalized_state.device)[:-1]
    values=(torch.bucketize(padded.contiguous(),bins,right=True)-1).cpu().tolist()
    return [f'Task: {t.strip().replace(chr(10), " ")}, State: {" ".join(map(str,s))};\nAction: ' for t,s in zip(tasks,values)]


class PromptCollator:
    """Tokenize immutable CPU state/tasks in existing bounded loader workers."""
    def __init__(self,root,max_length):
        self.root=str(root);self.max_length=max_length;self.tokenizer=None
        stats=json.loads((Path(root)/'statistics.json').read_text())['observation.state']
        self.offset=torch.tensor([(a+b)/2 for a,b in zip(stats['q01'],stats['q99'])],dtype=torch.float32)
        self.scale=torch.tensor([max((b-a)/2,1e-6) for a,b in zip(stats['q01'],stats['q99'])],dtype=torch.float32)
    def __call__(self,items):
        return self.prepare_batch(default_collate(items))
    def prepare_batch(self,batch):
        batch=dict(batch)
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            refs=json.loads((Path(self.root)/'checkpoints.json').read_text())
            self.tokenizer=AutoTokenizer.from_pretrained(refs['google/paligemma-3b-pt-224']['path'])
        prompts=state_prompt(batch['task'],(batch['observation.state']-self.offset)/self.scale)
        tokens=self.tokenizer(prompts,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
        batch['observation.language.tokens']=tokens.input_ids
        batch['observation.language.attention_mask']=tokens.attention_mask.bool()
        return batch


class CandidatePolicy(Policy):
    def prepare(self,batch):
        if 'observation.language.tokens' not in batch:return super().prepare(batch)
        device=self.velocity_mean.device
        b={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        correction=drift(b['velocity'],self.velocity_mean,20,b['offset']) if self.recipe['kr'] else torch.zeros_like(b['action'])
        model_batch={k:v for k,v in b.items() if k.startswith('observation.images.')}
        model_batch['observation.state']=self.affine(b['observation.state'],'observation.state')
        model_batch['action']=self.affine(coordinates(b['action'],correction),'action')
        model_batch['action_is_pad']=b['action_is_pad']
        model_batch['observation.language.tokens']=b['observation.language.tokens']
        model_batch['observation.language.attention_mask']=b['observation.language.attention_mask'].bool()
        model_batch['task']=b['task']
        return model_batch,correction


class TimedIterator:
    def __init__(self,it):self.it=iter(it);self.wait_seconds=0.;self.count=0
    def __iter__(self):return self
    def __next__(self):
        import time
        start=time.perf_counter();value=next(self.it)
        self.wait_seconds+=time.perf_counter()-start;self.count+=1
        return value


class CUDAPrefetch:
    """Keep native consumed cursor semantics: prefetch never advances progress."""
    def __init__(self,it,device):
        self.it=it;self.device=device;self.stream=torch.cuda.Stream(device=device);self.batch=None;self.finished=False
        self._preload()
    def _preload(self):
        try:cpu=next(self.it)
        except StopIteration:self.finished=True;self.batch=None;return
        with torch.cuda.stream(self.stream):
            self.batch={k:v.to(self.device,non_blocking=True) if torch.is_tensor(v) else v for k,v in cpu.items()}
    def __iter__(self):return self
    def __next__(self):
        if self.finished:raise StopIteration
        current=torch.cuda.current_stream(self.device);current.wait_stream(self.stream)
        batch=self.batch
        for value in batch.values():
            if torch.is_tensor(value) and value.is_cuda:value.record_stream(current)
        self._preload();return batch
