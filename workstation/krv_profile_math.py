"""Frozen causal profile displacement and its separate execution innovation."""
import numpy as np
from velocity_innovation import ball_clip

PROFILE_CAP=.05

def profile_extra(acceleration,tau):
    return ball_clip(.5*np.asarray(acceleration,dtype=np.float64)*float(tau)**2,PROFILE_CAP)

class ProfileInnovation:
    def __init__(self,enabled,dt):
        self.enabled=enabled;self.dt=float(dt);self.velocities=[];self.integrals=[]
        self.plan=None;self.offset=np.zeros(3);self.closed=False;self.rows=[]
    def observe(self,velocity):
        v=np.asarray(velocity,dtype=np.float64).reshape(3)
        self.integrals.append(self.integrals[-1]+.5*self.dt*(self.velocities[-1]+v) if self.velocities else np.zeros(3))
        self.velocities.append(v.copy())
    def set_plan(self,plan):
        assert 0<=plan['origin']<len(self.velocities)
        self.plan=plan
    def correct(self,action):
        base=np.asarray(action).copy();raw=np.zeros(3);age=0.;gate=False
        if self.plan is not None:
            if base[0,-1]<=0:self.closed=True
            origin=self.plan['origin'];age=(len(self.velocities)-1-origin)*self.dt
            planned=np.asarray(self.plan['velocity'])*age
            if self.plan['vpd']:planned+=profile_extra(self.plan['acceleration'],age)
            raw=self.integrals[-1]-self.integrals[origin]-planned
            gate=bool(self.plan['gate']) and not self.closed
        target=ball_clip(raw,.05) if self.enabled and gate else np.zeros(3)
        self.offset+=ball_clip(target-self.offset,.005)
        base[0,:3]+=self.offset
        self.rows.append({'index':len(self.velocities)-1,'plan':self.plan,'velocity':self.velocities[-1].tolist(),'age_s':age,'raw_innovation_m':raw.tolist(),'offset_m':self.offset.tolist(),'base_action':np.asarray(action).tolist(),'applied_action':base.tolist(),'gate':gate})
        return base
    def finish(self,folder):
        import json
        from pathlib import Path
        p=Path(folder)
        (p/'profile_correction.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in self.rows))
