"""Stage the unchanged real-data recipe for a two-V100 DOMv run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
PROJECT=Path(__file__).resolve().parents[1]
REMOTE=Path('/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def package(output):
    source=PROJECT/'scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114'
    c=json.loads((source/'launch_contract.json').read_text());c=copy.deepcopy(c)
    c.update(families=['domv'],local_verified_dataset=str(REMOTE/'dataset'),phase_cache=None,
        microbatch=2,gradient_accumulation=16,effective_batch_size=64,num_workers=4,
        precision='fp16_grad_scaler',parameter_dtype='float32',world_size=2,training_machine='Palmetto V100x2',
        sampling_weights=str(REMOTE/'sampling_weights.npy'),status='prepared',training_started=False)
    c['methods']={'domv':c['methods']['domv']};c['checkpoint']={'domv':c['checkpoint']['domv']}
    c['checkpoint_preferences'][0]['path']=str(REMOTE/'base')
    c['wandb']['run_names']={'domv':'palmetto-v100x2-fresh-domv'}
    c['wandb']['mode']='offline'  # No credentials copied from the workstation.
    c['source_hashes']={}
    scripts=['train_dom_posttraining_ddp.py','train_dom_posttraining.py','dom_posttraining.py','prepare_dom_posttraining.py','smoke_real_finetune.py']
    output.parent.mkdir(parents=True,exist_ok=True)
    contract_path=output.with_suffix('.contract.json');contract_path.write_text(json.dumps(c,indent=2)+'\n')
    def filter_file(info):
        return None if any(part in ['__pycache__','.git','.pytest_cache'] for part in Path(info.name).parts) or info.name.endswith('.pyc') else info
    with tarfile.open(output,'w',dereference=True) as archive:
        for name in scripts:archive.add(PROJECT/'workstation'/name,arcname='app/workstation/'+name)
        for name in ['real_robot','third_party/DynamicVLA']:
            archive.add(PROJECT/name,arcname='app/'+name,filter=filter_file)
        archive.add(PROJECT/'scratch/real_robot/finetuning-checkpoints/assets',arcname='app/scratch/real_robot/finetuning-checkpoints/assets')
        dataset=Path(json.loads((source/'launch_contract.json').read_text())['local_verified_dataset'])
        for name in ['data','meta','quality','videos']:archive.add(dataset/name,arcname='dataset/'+name,filter=filter_file)
        archive.add(source/'sampling_weights.npy',arcname='sampling_weights.npy')
        archive.add(source/'parquet_hashes.json',arcname='parquet_hashes.json')
        archive.add(contract_path,arcname='launch_contract.json')
        for name in ['config.json','model.safetensors']:
            archive.add(PROJECT/'scratch/real_robot/static-marker-eval/checkpoints/original-dom'/name,arcname='base/'+name)
    report=dict(archive=str(output),bytes=output.stat().st_size,sha256=sha(output),remote_root=str(REMOTE),
        original_base_weights_transferred=True,previous_finetuned_weights_transferred=False)
    output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);package(parser.parse_args().output.resolve())
