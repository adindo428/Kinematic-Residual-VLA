"""Focused validation-loader/model-startup diagnostic; no production writes."""
import argparse,json,os,sys,time,faulthandler
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
import torch
from core.dom_training import load_family,build_dataset,make_loader,build_policy,seed_all,to_device,amp_context,fixed_noise
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=0);p.add_argument('--microbatch',type=int,default=2);a=p.parse_args()
 root=Path(os.environ['DOM100_ROOT']);seed_all();faulthandler.enable()
 def stage(s):print(json.dumps({'stage':s,'time':time.time(),'workers':a.workers}),flush=True)
 m=load_family(root);stage('policy_build_before_validation')
 policy=build_policy(root,'domv100','cuda:0');stage('policy_ready')
 split=json.loads((root/'split.json').read_text())['splits']
 data=build_dataset(root,split['validation']);stage('validation_metadata_ready')
 runtime={'microbatch':a.microbatch,'workers':a.workers,'prefetch':2,'pin_memory':False,'fused':False,'precision':'bf16','sampling':'frames'}
 loader=make_loader(data,runtime,indices=json.loads((root/'validation_anchors.json').read_text())[:2*a.microbatch])
 policy.eval()
 for i,batch in enumerate(loader):
  stage('validation_batch_'+str(i));batch=to_device(batch,'cuda:0')
  with torch.no_grad(),amp_context('bf16','cuda:0'):
   policy.reset();prediction=policy.predict_action_chunk(batch,noise=fixed_noise(batch,policy,'cuda:0'))
  assert prediction.shape[1]==20 and torch.isfinite(prediction).all()
  torch.cuda.synchronize();stage('prediction_passed_'+str(i))
 stage('probe_passed')

if __name__=="__main__":main()
