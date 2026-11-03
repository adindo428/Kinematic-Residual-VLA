import torch
from safetensors import safe_open
from lerobot.configs.types import PolicyFeature,FeatureType
from vlash.policies.pi05.configuration_pi05 import PI05Config
from vlash.policies.pi05.modeling_pi05 import PI05Model
c=PI05Config();c.state_cond=False;c.fuse_qkv=c.fuse_gate_up=False
with torch.device('meta'):m=PI05Model(c)
with safe_open('/scratch/aparame/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30/model.safetensors',framework='pt') as f:
 keys=list(f.keys());print('FILE SAMPLE',keys[:6]);print('TARGET SAMPLE',list(m.state_dict())[:6])
 print('FILECOUNT',len(keys),'TARGETCOUNT',len(m.state_dict()))
