from pathlib import Path
import sys
import shlex
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from palmetto_cpu import command
from train_dom_posttraining_ddp import DistributedEpochSampler
from dom_posttraining import accumulation_size
from real_robot.policy_precision import apply_checkpoint_precision


def test_cpu_transport_only_runs_srun_on_login_and_quotes_the_entire_cpu_body():
    body='printf "%s" "literal $(hostname)"\npython3 train.py'
    cmd=command(body);remote=shlex.split(cmd[-1])
    assert remote[:2]==['exec','srun']
    assert '--jobid=20145158' in remote and '--nodelist=node1347' in remote
    assert remote[-3:-1]==['bash','-lc'] and remote[-1].endswith(body)
    assert 'SLURM_JOB_ID' in remote[-1] and 'hostname -s' in remote[-1]
    assert 'exec </dev/null' in remote[-1]
    with pytest.raises(ValueError,match='DTN'):command('x'*100001)
    for node in ['slogin','slogin.palmetto.clemson.edu','node1347; ls']:
        with pytest.raises(ValueError):command(body,node=node)


def test_distributed_weighted_draw_is_reproducible_and_pads_only_for_equal_microbatches():
    weights=np.arange(1,102,dtype=float)
    samplers=[DistributedEpochSampler(weights,rank,2,4) for rank in range(2)]
    for sampler in samplers:sampler.epoch=9
    draws=[list(sampler) for sampler in samplers]
    assert len(draws[0])==len(draws[1])==52 and samplers[0].total==104
    assert list(samplers[0])==draws[0] and draws[0]!=draws[1]
    expected=torch.multinomial(torch.tensor(weights),104,True,generator=torch.Generator().manual_seed(51)).tolist()
    interleaved=[row[1] for pair in zip(*draws) for row in pair]
    assert interleaved==expected
    assert all(epoch==9 for draw in draws for epoch,index in draw)


def test_two_rank_accumulation_matches_global_mean_at_the_short_epoch_tail():
    # 104 global draws -> 52 per rank -> six full windows plus a four-sample tail.
    inputs=torch.arange(1.,105.).reshape(52,2).T
    actual=[]
    for rank in range(2):
        param=torch.tensor(.2,requires_grad=True)
        local=inputs[rank,-4:]
        loss=(param*local-1).square().mean()
        (loss*len(local)/accumulation_size(12,4,2,52)).backward()
        actual.append(param.grad)
    reference=torch.tensor(.2,requires_grad=True)
    (reference*inputs[:,-4:]-1).square().mean().backward()
    torch.testing.assert_close(torch.stack(actual).mean(),reference.grad)


def test_fp32_v100_checkpoint_loads_without_reducing_saved_master_precision(tmp_path):
    import json
    source=tmp_path/'epoch-001';source.mkdir()
    (source/'provenance.json').write_text(json.dumps(dict(parameter_dtype='float32',autocast_dtype='float16')))
    selected=tmp_path/'selected';selected.mkdir()
    (selected/'selection.json').write_text(json.dumps(dict(source=str(source))))
    policy=torch.nn.Linear(3,2).bfloat16()
    apply_checkpoint_precision(policy,selected)
    assert policy.weight.dtype==torch.float32 and policy.execution_autocast_dtype==torch.float16
    # An older checkpoint retains its existing native mixed parameter dtypes.
    legacy=tmp_path/'legacy';legacy.mkdir();policy=torch.nn.Linear(3,2).bfloat16()
    apply_checkpoint_precision(policy,legacy)
    assert policy.weight.dtype==torch.bfloat16 and policy.execution_autocast_dtype==torch.bfloat16
