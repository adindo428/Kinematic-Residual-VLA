"""CPU contracts and real train/validation decode before GPU allocation."""
import json,sys,math
from pathlib import Path
R=Path('/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029')
sys.path.insert(0,str(R/'source'))
from utils.dom_experiment import digest,atomic_json,VARIANTS
from core.dom_training import load_family,build_dataset,make_scheduler
from core.dom_distributed import RankBatchSampler
import torch
m=load_family(R/'family');p=json.load(open(R/'family/recipe.json'));runtime=json.load(open(R/'runtime.json'))
split=json.load(open(R/'family/split.json'))['splits'];tests=[]
def check(ok,name):assert ok,name;tests.append(name)
check(VARIANTS['krv100']==(True,True,'constant'),'native analytic residual plus observed velocity; constant gate')
check([len(split[k]) for k in ('train','validation','test')]==[20000,2015,2011],'exact split counts')
check(not(set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test'])),'disjoint episode membership')
check(m['split_sha256']=='98436815f85fbfc78d48e4332e8603684f6bae1620400b181767bab1c5736d88','exact PC split hash')
check(m['weights_sha256']=='e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8','released initialization pin')
check(p['epochs']==10 and not p['early_stopping']['enabled'] and p['lr']==3e-5,'ten full epochs; fresh native recipe')
check(math.ceil(m['train_frames']/352)*10==68380,'ceil budget includes final149-frame batch')
# Exercise native sharding on odd tail and exact mid-epoch resume.
for cursor in (0,352):
 ranks=[list(RankBatchSampler([705],32,r,2,cursor=cursor)) for r in range(2)]
 check(len(ranks[0])==len(ranks[1]),'equal rank collective counts at cursor'+str(cursor))
 seen=[i[1] for batches in ranks for batch in batches for i in batch if i[2]]
 check(len(seen)==705-cursor and len(set(seen))==len(seen),'full tail ownership at cursor'+str(cursor))
param=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([param],lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10);sched=make_scheduler(opt,68380)
check(opt.param_groups[0]['lr']==0,'fresh warmup starts at zero')
for _ in range(1001):param.grad=torch.ones_like(param);opt.step();sched.step()
state=opt.state_dict();sch=sched.state_dict();lr=opt.param_groups[0]['lr'];o=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=3e-5);s=make_scheduler(o,68380);o.load_state_dict(state);s.load_state_dict(sch)
check(s.last_epoch==1001 and o.param_groups[0]['lr']==lr,'optimizer scheduler strict continuation')
for part in ('train','validation'):
 data=build_dataset(R/'family',[split[part][0]],training=part=='train');sample=data[(0,0)]
 check(all(torch.isfinite(v).all() for v in sample.values() if torch.is_tensor(v)),'decoded finite '+part+' two-camera anchor')
 check(all(k in sample for k in ('observation.images.opst_cam','observation.images.wrist_cam','action')),'decoded '+part+' required fields')
 del data,sample
atomic_json(R/'cpu-preflight.json',{'passed':True,'tests':tests,'count':len(tests),'manifest':m['fingerprint'],'source':m['source_sha256'],'runtime':runtime})
print(json.dumps({'passed':len(tests),'source':m['source_sha256']}),flush=True)
