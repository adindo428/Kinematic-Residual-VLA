"""Prepare upstream-only SmolVLA on the frozen static-orange 80/10/10 split."""
import hashlib,json
from pathlib import Path
import numpy as np
from family_data import RealFamilyDataset,prepare_statistics

PROJECT=Path(__file__).resolve().parents[1]


def main():
    root=PROJECT/'scratch/real_robot/finetuning-runs/static-orange-smolvla-pretrained-20261113'
    parent=PROJECT/'scratch/real_robot/finetuning-runs/static-orange-krv-overfit80-20261113'
    c=json.loads((parent/'launch_contract.json').read_text());source=json.loads((root/'upstream_source.json').read_text());assert source['verified']
    legacy=PROJECT/'scratch/real_robot/family-finetuning-v6'
    for name in ['hub','dependencies']:
        if not (root/name).exists():(root/name).symlink_to(legacy/name,target_is_directory=True)
    c.update(families=['smolvla_baseline'],sampling='uniform_permutation',instruction_mode='recorded',
        final_evaluation_episodes=json.loads((Path(c['local_verified_dataset'])/'meta/splits.json').read_text())['validation'])
    for key in ['checkpoint','checkpoint_preferences','initialization_checkpoint','methods','throughput_benchmark','benchmark_reuse_note','batch_profile']:c.pop(key,None)
    c['overfit_diagnostic']['purpose']='Original upstream SmolVLA fine-tuned only on 80 static-orange training episodes; 10 validation and 10 test preserved'
    c['overfit_diagnostic']['split']='original';c['overfit_diagnostic']['training_episodes']=80;c['overfit_diagnostic']['independent_holdout']=True
    c['wandb']['run_names']={'smolvla_baseline':'static-orange-smolvla-upstream-only'}
    entry=dict(family='smolvla_baseline',upstream_pretrained=source,velocity_conditioning=None,
        checkpoint=dict(id='lerobot/smolvla_base@'+source['revision'],code='code/adapters/smolvla_baseline',format='upstream_safetensors'),sha256=source['weight_sha256'])
    campaign=dict(repo='lerobot/smolvla_base',revision=source['revision'],entries=[entry],training_recipe=c,
        run_prefix='static-orange-smolvla-original-pretrained',ema=dict(decay=.999),benchmark_before_training=False,
        family_performance={'smolvla_baseline':dict(flow_microbatch=8,auxiliary_microbatch=2,loader_workers=4,prefetch_factor=2,vectorized_photometric=False,decoder_cache_episodes=8)},
        initialization=dict(dom_weights_loaded=False,real_robot_weights_loaded=False,upstream_only=True),
        evaluation_instruction='Pick up the rolling orange and place it in the box.')
    data=RealFamilyDataset(campaign,'smolvla_baseline');weights=np.ones(len(data),dtype=np.float64);np.save(root/'sampling_weights.npy',weights)
    c.update(sampling_weights=str(root/'sampling_weights.npy'),sampling_weights_sha256=hashlib.sha256((root/'sampling_weights.npy').read_bytes()).hexdigest())
    probe=json.loads((parent/'krv100/action_validation_indices.json').read_text())['indices'];validation=RealFamilyDataset(campaign,'smolvla_baseline','validation')
    assert len(data)==12564 and len(validation)==1110 and max(probe)<len(validation)
    c['action_validation_indices']=probe
    (root/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    (root/'downloads.json').write_text(json.dumps(dict(models={'smolvla_baseline':dict(state='verified',sha256=source['weight_sha256'])}),indent=2)+'\n')
    stats=prepare_statistics(campaign,root/'real_statistics.json')
    print(json.dumps(dict(train_anchors=len(data),validation_anchors=len(validation),sample_prompt=data[0]['task'],initialization=source),indent=2))


if __name__=='__main__':main()
