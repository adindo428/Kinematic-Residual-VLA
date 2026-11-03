import hashlib, json
from importlib.metadata import version
from pathlib import Path

root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
setup=root/'setup-20261008'
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261008T201101Z')
inventory=json.loads((setup/'download-inventory.json').read_text())
report={'work_root':str(root),'code_snapshot':str(source),'environment':str(root/'venv-krvla-dom-20261008'),'module':'python/miniforge3_pytorch/2.12.0','slurm_account':'bivv-dtai-gh','partition':'ghx4','activation':str(setup/'activate-training.sh'),'assets':[],'training_launched':False}
for record in inventory:
    name=record['repo'].replace('/','--')
    status=json.loads((setup/(name+'-status.json')).read_text())
    report['assets'].append(status)
normalizers=root/'tracking/normalization-reference/model.best_validation_action_mse.safetensors'
report['tracking_normalizers']={'path':str(normalizers),'sha256':hashlib.sha256(normalizers.read_bytes()).hexdigest(),'source':'/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_baseline_16623336/model.best_validation_action_mse.safetensors','selection':'Only normalize_inputs, normalize_targets, unnormalize_outputs tensors'}
config_dest=setup/'tracking-configs'; config_dest.mkdir(exist_ok=True)
configs=[]
for path in (source/'tracking/configs').glob('*.yaml'):
    if path.name.startswith('tracking_dom75_full20_') or path.name=='dynamicvla_dom_tracking_velocity75.yaml':
        text=path.read_text()
        text=text.replace('/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_baseline_16623336',str(normalizers.parent))
        text=text.replace('configs/dom_franka_tracking_split.json',str(source/'tracking/configs/dom_franka_tracking_split.json'))
        text=text.replace('CHECKPOINT: hzxie/dynamic-vla-DOM','CHECKPOINT: '+str(root/'dom100/initialization'))
        (config_dest/path.name).write_text(text)
        configs.append(str(config_dest/path.name))
report['tracking_configs']=configs
report['tracking_initialization_argument']=['--init-ckpt',str(root/'dom100/initialization')]
report['remaining_before_training']=['Complete the download of files referenced by the matched Palmetto split','Profile GH200 memory and configure training launchers']
if (setup/'environment-report.json').is_file():
    report['environment_validation']=json.loads((setup/'environment-report.json').read_text())
    report['environment_validation']['packages']={p:version(p) for p in ('torch','torchvision','torchcodec','av','transformers','huggingface-hub','lerobot','pylibmc')}
report['native_packages']=[{k:json.loads(p.read_text()).get(k) for k in ('name','version','build')} for p in (root/'native-krvla-dom/conda-meta').glob('*.json')]
report['dom_download_status_file']=str(setup/'hzxie--DOM-status.json')
report['dom_download_log']=str(setup/'hzxie--DOM-download.log')
report['dom_training_family']=str(root/'dom100-palmetto-h20')
report['reused_preparation']=json.loads((root/'dom100-palmetto-h20/REUSE_REPORT.json').read_text())
(setup/'SETUP_REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
