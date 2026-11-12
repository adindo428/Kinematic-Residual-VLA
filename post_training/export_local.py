"""Export a private, exact protocol bundle locally. Never upload it to GitHub."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

PROJECT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def export(campaign_path,output,video_revision):
    campaign_path=campaign_path.resolve();source=campaign_path.parent
    output=output.resolve()
    if output.exists():raise ValueError('Use a fresh output directory; do not overwrite an existing private bundle')
    if not video_revision or len(video_revision)!=40:raise ValueError('Supply a full pinned video dataset commit')
    campaign=read(campaign_path);dataset=Path(campaign['training_recipe']['local_verified_dataset'])
    output.mkdir(parents=True)
    shutil.copyfile(campaign_path,output/'campaign.v6.json')
    for name in ['real_statistics.json','dependency-patch-lock.json','transformers-replacement-source.json','benchmark-summary.json']:
        shutil.copyfile(source/name,output/name)
    shutil.copyfile(campaign['training_recipe']['sampling_weights'],output/'sampling_weights.npy')
    reference=output/'reference-benchmarks';reference.mkdir()
    for entry in campaign['entries']:
        file=source/'benchmarks'/f"{entry['family']}.json"
        report=read(file)
        if report.get('passed') and report.get('state')=='completed':shutil.copyfile(file,reference/file.name)
    # Locate the same original ten-episode list used by the source trainer.
    original_project=Path(campaign['training_recipe']['local_verified_dataset']).parents[3]
    summary=original_project/'scratch/real_robot/action-evaluation/validation-10-seed12345/summary.json'
    if not summary.is_file():raise ValueError('Original ten-episode evaluation summary is missing')
    shutil.copyfile(summary,output/'validation-10-summary.json')
    completion=Path(campaign['current_queue_dependency'])/'final-evaluation-status.json'
    if read(completion).get('state')!='completed':raise ValueError('Prior campaign has not completed')
    shutil.copyfile(completion,output/'prior-campaign-completion.json')
    shutil.copyfile(source/'hub/files.json',output/'hub-files.json')
    # Required runtime view only, keeping all label/quality bytes unchanged.
    meta=['info.json','splits.json','quality_policy.json','episodes.jsonl','tasks.jsonl',
          'train_stats.json','image_preprocessing.json','grasp_annotations.json']
    paths=sorted([*dataset.glob('data/**/*.parquet'),*dataset.glob('quality/*.jsonl'),
                  *[dataset/'meta'/name for name in meta]])
    labels={str(p.relative_to(dataset)):sha(p) for p in paths}
    archive=output/'dataset-labels.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path in paths:
            info=tar.gettarinfo(str(path),arcname=str(path.relative_to(dataset)))
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with path.open('rb') as f:tar.addfile(info,f)
    videos={}
    for camera in ['observation.images.opst_cam','observation.images.wrist_cam']:
        for path in sorted((dataset/'videos').rglob(camera+'/episode_*.mp4')):
            videos[str(path.relative_to(dataset))]=sha(path)
    write(output/'dataset-manifest.json',dict(schema=1,repo_id=campaign['training_recipe']['repo_id'],
        revision=video_revision,label_archive_sha256=sha(archive),labels=labels,videos=videos,
        source_quality_policy_sha256=campaign['training_recipe']['dataset_quality_policy_sha256']))
    # Virtual paths let this private directory live outside the Git checkout.
    locked={}
    for p in sorted((PROJECT/'post_training').rglob('*')):
        if p.is_file() and 'frozen' not in p.relative_to(PROJECT/'post_training').parts and '__pycache__' not in p.parts:
            locked[str(p.relative_to(PROJECT))]=sha(p)
    for p in (PROJECT/'workstation').glob('*'):
        if p.is_file() and p.name.startswith(('family','train_family','profile_family','run_family','run_benchmarked',
                                               'download_family','velocity_conditioning','finetune-python')):
            locked[str(p.relative_to(PROJECT))]=sha(p)
    for name in ['real_robot/action_objective.py','real_robot/training_augmentation.py',
                 'real_robot/fast_training_augmentation.py','real_robot/training_ema.py',
                 'third_party/DynamicVLA/utils/real_robot_contract.py','third_party/DynamicVLA/utils/instruction_generator.py']:
        locked[name]=sha(PROJECT/name)
    for p in output.rglob('*'):
        if p.is_file():locked['post_training/frozen/'+str(p.relative_to(output))]=sha(p)
    write(output/'protocol-lock.json',dict(schema=1,metric_source_campaign='family-finetuning-v6',files=dict(sorted(locked.items()))))
    print('Private protocol bundle:',output)
    print('Transfer directly to the other computer; do not commit or publish it.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,default=PROJECT/'post_training/frozen')
    p.add_argument('--video-revision',required=True)
    a=p.parse_args();export(a.source_campaign,a.output,a.video_revision)


if __name__=='__main__':main()
