#!/usr/bin/env python3
"""Stage the user's retained checkpoints and reproducibility files for HF."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
WORK = REPO/'scratch/dom45'
ROOT = WORK/'releases/kinematic-residual-vla-checkpoints-20261108'
STAGE = ROOT/'staging'
REPO_ID = 'adi2440/kinematic-residual-vla-checkpoints'
FILES = {}; MODELS = []; ALIASES = {}
SENSITIVE = re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def add(src, dst, binary=False):
    src = Path(src).resolve(); target = STAGE/dst
    target.parent.mkdir(parents=True, exist_ok=True)
    if not binary:
        data = src.read_bytes()
        if SENSITIVE.search(data): raise ValueError('Credential-like literal in public file: '+str(src))
        if not target.exists() or target.read_bytes() != data: target.write_bytes(data)
    elif not target.exists(): os.link(src, target)
    assert target.stat().st_size == src.stat().st_size
    FILES[str(target.relative_to(STAGE))] = {'source': str(src), 'bytes': src.stat().st_size,
                                           'mtime_ns': src.stat().st_mtime_ns, 'binary': binary}


def code_tree(src, dst):
    src = Path(src)
    for path in src.rglob('*'):
        if not path.is_file() or any(p in {'.git', '__pycache__', 'wandb', 'outputs', '.cache'} for p in path.relative_to(src).parts): continue
        if path.suffix not in {'.py', '.json', '.yaml', '.yml', '.toml', '.txt', '.model', '.merges'} and path.name not in {'LICENSE', 'NOTICE'}: continue
        if path.stat().st_size > 32*1024**2: continue
        add(path, str(Path(dst)/path.relative_to(src)))


def family_metadata(family, dst):
    for path in family.glob('*.json'): add(path, str(Path(dst)/path.name))
    for path in (family/'runs').glob('*/*.json'):
        if not path.name.startswith(('epoch-', 'validation-', 'evaluation-')) and path.name not in {'recipe.json','provenance.json','best.json','latest.json','final.json','training_complete.json'}: continue
        add(path, str(Path(dst)/'run_records'/path.parent.name/path.name))


def native(directory, scope, model, code, family=None, aliases=(), resume=None, selected=None):
    import torch
    directory = Path(directory); resume = Path(resume) if resume else directory/'training_state.pt'
    assert resume.is_file(), 'Missing training state: '+str(resume)
    state = torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
    progress = state['progress']; del state
    cfg=json.loads((directory/'config.json').read_text())
    assert cfg.get('completed_epochs',cfg.get('epoch'))==progress['epoch'],'Weights/config and optimizer generation differ'
    epoch = int(progress['epoch']); updates = int(progress['updates']); partial = bool(progress.get('cursor'))
    leaf = f'epoch-{epoch:03d}'+('-partial-next' if partial else '')+f'-u{updates:09d}'
    dst = f'checkpoints/{scope}/{model}/{leaf}'
    for name in ['model.safetensors','config.json','provenance.json','checkpoint_metadata.json']:
        if (directory/name).exists(): add(directory/name, f'{dst}/{name}', binary=name.endswith('.safetensors'))
    add(resume, f'{dst}/training_state.pt', binary=True)
    key = f'{model}-{scope.replace("/", "-")}-{leaf}'
    item = {'id': key, 'path': dst, 'format': 'native_safetensors', 'model': model, 'scope': scope,
            'progress': progress, 'code': code, 'family': family, 'optimizer_state': True,
            'training_state': f'{dst}/training_state.pt', 'selection': selected,
            'aliases': list(aliases), 'partial_next_epoch': partial}
    MODELS.append(item)
    for alias in aliases: ALIASES[alias] = key
    write(STAGE/dst/'release.json', item)


def adapters():
    import torch
    catalog = json.loads((WORK/'evaluations/low-clutter-full-table1-20261102/catalog.json').read_text())
    old = json.loads((WORK/'evaluations/multivla20k-20261029/catalog.json').read_text())
    paths = []
    for name in ['palmetto-vla-20261026','palmetto-updated-20261102']:
        paths += sorted((WORK/'training'/name).rglob('latest.pt'))
    for path in paths:
        name = path.parent.name
        if name == 'pi0.5kr': name = 'pi05_kr'
        c = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        progress, identity = c['progress'], c['identity']; keys = set(c); del c
        assert {'model','optimizer','scheduler','rng'} <= keys
        scope = '20k-continuation'; leaf = f'epoch-{progress["epoch"]:03d}'+('-partial-next' if progress['cursor'] else '')+f'-u{progress["updates"]:09d}'
        dst = f'checkpoints/{scope}/{name}/{leaf}'; add(path, f'{dst}/checkpoint.pt', binary=True)
        entry = catalog['models'].get(name, old['models'].get(name))
        code = f'code/adapters/{name}'; source = WORK/'evaluations/multivla20k-20261029/model_code'/name
        assert source.is_dir(); code_tree(source, code)
        aliases = []
        if entry and Path(entry['checkpoint']).resolve() == path.resolve(): aliases = [name+'-20k-latest']
        elif name == 'vlash_baseline' and 'palmetto-updated' in str(path): aliases = [name+'-20k-latest-incomplete']
        key = f'{name}-{scope}-{leaf}'
        exposure = json.loads((source/'data-provenance.json').read_text())
        item = {'id': key, 'path': dst, 'format': 'adapter_torch', 'model': name, 'scope': scope,
                'progress': progress, 'identity': identity, 'code': code,
                'training_state': f'{dst}/checkpoint.pt', 'optimizer_state': True,
                'selection': 'latest transferred snapshot at these updates; validation-best unverified',
                'partial_next_epoch': bool(progress['cursor']), 'aliases': aliases,
                'prior_full_data_exposure': bool(exposure.get('prior_full_data_exposure'))}
        MODELS.append(item)
        for alias in aliases: ALIASES[alias] = key
        write(STAGE/dst/'release.json', item)
        for meta in path.parent.glob('*.json'):
            if meta.stat().st_size < 2*1024**2: add(meta, f'{dst}/original_run_records/{meta.name}')
    for name in {m['model'] for m in MODELS if m['format']=='adapter_torch'}:
        current=max((m for m in MODELS if m['format']=='adapter_torch' and m['model']==name),key=lambda m:m['progress']['updates'])
        alias=name+'-20k-latest';ALIASES[alias]=current['id']
        for item in MODELS:
            if item['format']=='adapter_torch' and item['model']==name:
                item['aliases']=[a for a in item['aliases'] if a!=alias]
                if item is current:item['aliases'].append(alias)
                write(STAGE/item['path']/'release.json',item)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    base = WORK/'training/krv100-full-20261010/source'
    for name, overlay in [('native-dom', None), ('native-krv-full', WORK/'training/krv100-full-deltaai-20261106/source'),
                          ('native-krv-20k', WORK/'training/krv100-fresh20k-h200-20261029/source')]:
        code_tree(base, 'code/'+name)
        if overlay: code_tree(overlay, 'code/'+name)
    full = [
      ('krv100-full-deltaai-20261106/best-epoch003', 'krv100', 'native-krv-full', 'krv100-epoch003', ['krv100-full-best'], 'validation-selected epoch 3'),
      ('krv100-full-deltaai-20261106/latest-after-epoch004-u000304483', 'krv100', 'native-krv-full', 'krv100-after-epoch004-partial005', ['krv100-full-latest'], '4 completed epochs plus partial epoch 5'),
      ('domv-full-deltaai-20261106/best-epoch003', 'domv', 'native-dom', 'domv-epoch003', ['domv-large-best'], 'validation-selected epoch 3; audited available-data split, 197198 train episodes')]
    for directory, model, code, resume, aliases, selected in full:
        scope='large-audited-197198' if model=='domv' else 'full'
        native(WORK/'training'/directory, scope, model, 'code/'+code, family=f'training/{scope}/{model}',aliases=aliases,
               resume=ROOT/'remote_states'/resume/'training_state.pt', selected=selected)
    for model in ['krv100','domv']:
        scope='large-audited-197198' if model=='domv' else 'full'
        family_metadata(ROOT/'remote_states'/f'family-full-{model}',f'training/{scope}/{model}')
    for generation in sorted((ROOT/'remote_states/domv-generations').glob('u*')):
        aliases=['domv-large-latest','domv-large-final'] if generation.name.startswith('u000381312') else []
        native(generation,'large-audited-197198','domv','code/native-dom',family='training/large-audited-197198/domv',aliases=aliases,
               selected='final; early stopped at completed epoch 6' if aliases else 'retained full-data milestone')
    families = [
      ('fresh20k-20261025', 'dom100', '20k-fresh', 'dom', 'native-dom'),
      ('krv100-fresh20k-h200-20261029', 'krv100', '20k-fresh', 'krv100', 'native-krv-20k'),
      ('krv100-20k-from-epoch003-20261021', 'krv100', 'mixed-full-then-20k', 'krv100', 'native-dom'),
      ('krv100-full-20261010', 'krv100', 'full-historical', 'krv100', 'native-dom')]
    for folder, variant, scope, model, code in families:
        family = WORK/'training'/folder/'family'; meta = f'training/{scope}/{model}'
        family_metadata(family, meta)
        run = family/'runs'/variant; pointers = {}
        for name in ['best','latest','final']:
            path = run/(name+'.json')
            if path.exists(): pointers[name] = json.loads(path.read_text())['generation']
        for generation in sorted((run/'generations').iterdir()):
            if not (generation/'model.safetensors').exists() or generation.name.startswith('.'): continue
            aliases = [f'{model}-{scope}-{name}' for name, g in pointers.items() if g == generation.name]
            native(generation, scope, model, 'code/'+code, meta, aliases=aliases,
                   selected=', '.join(name for name,g in pointers.items() if g==generation.name) or 'retained milestone')
    adapters()
    source = WORK/'training/krv100-20k-from-epoch003-20261021/github-source/experiments/main8_20k_10epochs'
    for path in source.glob('*.py'): add(path, 'code/adapter-trainer/'+path.name)
    for name in ['split.json','selected-episodes.json','subset-report.json','statistics.json','asset-sizes.json']:
        if (source/name).exists(): add(source/name, 'training/20k-adapters/'+name)
    code_tree(WORK/'evaluations/multivla20k-20261029/vlash_source', 'code/vlash')
    code_tree(WORK/'evaluations/multivla20k-20261029/metadata', 'assets')
    native_assets=WORK/'cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots'
    for snapshot in sorted(native_assets.iterdir()):
        code_tree(snapshot, 'assets/HuggingFaceTB/SmolLM2-360M')
    for source in [WORK/'training/fresh20k-20261025/controller', WORK/'training/krv100-fresh20k-h200-20261029/scripts']:
        code_tree(source, 'code/training-controls/'+source.parent.name)
    add(base/'LICENSE', 'licenses/S-Lab-LICENSE')
    for path in (ROOT/'license_sources').glob('*'): add(path, 'licenses/'+path.name)
    for name in ['smolvla_table1_client.py','completed_vla_table1_client.py']:
        add(REPO/'workstation'/name, 'code/evaluation/'+name)
    geometry=next((WORK/'evaluations/multivla20k-20261029').rglob('rpc_geometry.py'))
    add(geometry, 'code/evaluation/rpc_geometry.py')
    add(WORK/'evaluations/multivla20k-20261029/workstation/multivla_table1_client.py', 'code/evaluation/multivla_table1_client.py')
    add(REPO/'workstation/hf_release_checkpoint_tools.py', 'tools/checkpoint_tools.py')
    packages_to_record=['torch','torchvision','transformers','lerobot','safetensors','huggingface_hub','numpy','pyarrow','timm','accelerate','tokenizers','einops','draccus','av','torchcodec','wandb','omegaconf','pandas','sentencepiece','pillow']
    for flavor,environment in [('native','train'),('adapters','eval20k')]:
        script='import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in '+repr(packages_to_record)+' if m.packages_distributions().get(n.replace("-","_")) or n in {d.metadata["Name"] for d in m.distributions()}}))'
        recorded=json.loads(subprocess.check_output([str(WORK/'envs'/environment/'bin/python'),'-c',script],text=True))
        write(STAGE/f'environment-{flavor}.json',recorded)
        requirements=['# Install CUDA-compatible PyTorch for your machine first.', '# LeRobot is bundled under code/runtime-* to preserve the exact implementation.']+[f'{name}=={ver.split("+")[0]}' for name,ver in recorded.items() if name not in {'torch','torchvision','torchcodec','lerobot'}]
        (STAGE/f'requirements-{flavor}.txt').write_text('\n'.join(requirements)+'\n')
        lib=WORK/'envs'/environment/'lib/python3.10/site-packages'
        code_tree(lib/'lerobot',f'code/runtime-{flavor}/lerobot')
        for path in lib.glob('lerobot-*.dist-info/licenses/*'):
            if path.is_file():add(path,f'licenses/lerobot-{flavor}-{path.name}')
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ['torch','torchvision','transformers','lerobot','safetensors','huggingface_hub','numpy','pyarrow','timm','accelerate','tokenizers','einops','draccus','av','torchcodec']:
        try: packages[name] = version(name)
        except PackageNotFoundError: pass
    write(STAGE/'environment_versions.json', packages)
    for item in MODELS:
        family=item.get('family')
        manifest=STAGE/family/'manifest.json' if family else None
        if manifest and manifest.exists():
            data=json.loads(manifest.read_text());item['dataset_counts']=data.get('counts')
            item['dataset_selection_mode']=data.get('dataset_selection_mode')
            item['dataset_revision']=data.get('dataset_revision')
            write(STAGE/item['path']/'release.json',item)
    write(STAGE/'model-index.json', {'title': 'Kinematic Residual VLA Checkpoints', 'repo_id': REPO_ID,
          'created_utc': datetime.now(timezone.utc).isoformat(), 'checkpoints': MODELS, 'aliases': ALIASES,
          'excluded': ['untrained initialization copies','smoke/profiling/debug outputs','duplicate evaluation exports',
                       'retired/deleted mixed epoch-3 weights, which are unavailable']})
    write(ROOT/'local_sources.private.json', FILES)
    total = sum(x['bytes'] for x in FILES.values())
    write(ROOT/'preparation.json', {'prepared': True, 'models': len(MODELS), 'files': len(FILES), 'logical_bytes': total,
          'repo_id': REPO_ID, 'public_requested': True, 'account_confirmed': 'adi2440'})
    print(json.dumps({'models': len(MODELS), 'files': len(FILES), 'logical_GB': total/1e9}))


if __name__ == '__main__': main()
