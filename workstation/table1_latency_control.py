"""Cooperative phase-boundary resource control for the evaluation controller."""
import threading
import time

import run_low_clutter_table1 as original


class ModelHeld(RuntimeError):
 """This model is deferred; other pending checkpoints may still run."""


class LatencyGate:
 def __init__(self,catalog,stop,status):
  self.catalog=catalog;self.stop=stop;self.status=status
  self.condition=threading.Condition();self.active=set();self.qualifying=False;self.pending=False
  self.qualified=False;self.held=set();self.limit=2;self.mode='parallel_shared'

 def check_request(self):
  path=original.ROOT/'runtime_health/mode_request.json'
  if path.exists() and original.read(path).get('mode')=='serial':
   with self.condition:
    self.limit=1;self.mode='adaptive_serial';self.condition.notify_all()

 def enter(self,lane,name,dimension):
  heavy=name in ['vlash_kr','pi05_baseline']
  if heavy:
   with self.condition:
    if not self.qualified:
     self.pending=True
     self.status(lane,state='waiting_for_latency_qualification',model=name,phase='qualification',dimension=None)
     while (self.active or self.qualifying) and not self.qualified:
      if self.stop.is_set():raise InterruptedError('Stopped while waiting for latency qualification')
      self.condition.wait(1)
     if not self.qualified:self.qualifying=True
   if self.qualifying and not self.qualified:
    try:
     from qualify_table1_heavy_latency import qualify
     result=qualify(self.catalog,self.stop)
     with self.condition:
      self.held=set(result.get('held_models',['vlash_kr','pi05_baseline'] if result['mode']=='hold_heavy' else []))
      if result['mode']=='serial':self.limit=1;self.mode='adaptive_serial'
      self.qualified=True
    finally:
     with self.condition:
      self.qualifying=False;self.pending=False;self.condition.notify_all()
  self.check_request()
  with self.condition:
   if name in self.held:
    self.status(lane,state='held_for_latency_review',model=name,phase='qualification',dimension=dimension)
    raise ModelHeld(name+' is held by the isolated latency qualification; continue other queued models.')
   while len(self.active)>=self.limit or self.pending or self.qualifying:
    self.status(lane,state='waiting_for_lane',model=name,phase='waiting',dimension=dimension)
    if self.stop.is_set():raise InterruptedError('Stopped at phase boundary')
    self.condition.wait(1)
   if self.stop.is_set():raise InterruptedError('Stopped at phase boundary')
   self.active.add(lane)

 def leave(self,lane):
  with self.condition:
   self.active.discard(lane);self.condition.notify_all()
