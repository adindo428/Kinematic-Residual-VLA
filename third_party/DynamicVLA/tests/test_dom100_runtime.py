import ast
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

from core.dom_checkpoint import load_generation, resolve_generation, run_lock, save_generation
from core.dom_training import make_scheduler
from utils.dom_data import DOMData, EpochBatchSampler, sample_seed
from utils.dom_experiment import VARIANTS, grouped_stratified_split, select_recipe


@pytest.mark.parametrize("sampling", ["frames", "blocks"])
def test_sampler_visits_every_frame_once_and_resume_ignores_prefetch(sampling):
    lengths = [3, 21, 9, 37]
    flatten = lambda s: [item for batch in s for item in batch]
    whole = flatten(EpochBatchSampler(lengths, 4, epoch=7, sampling=sampling))
    assert sorted(idx for _, idx in whole) == list(range(sum(lengths)))
    assert whole == flatten(EpochBatchSampler(lengths, 4, epoch=7, sampling=sampling))
    resumed = flatten(EpochBatchSampler(lengths, 4, epoch=7, cursor=20, sampling=sampling))
    assert resumed == whole[20:]
    assert whole != flatten(EpochBatchSampler(lengths, 4, epoch=8, sampling=sampling))
    assert list(EpochBatchSampler(lengths, 4, cursor=sum(lengths), sampling=sampling)) == []


def test_sample_augmentation_randomness_replays_without_changing_training_rng():
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    def draw():
        with sample_seed(987654321):
            return random.random(), np.random.rand(), torch.randn(3)
    a, b = draw(), draw()
    assert a[:2] == b[:2]
    torch.testing.assert_close(a[2], b[2])
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    torch.testing.assert_close(torch.get_rng_state(), before[2])


def test_grouped_split_is_deterministic_stratified_and_keeps_duplicate_demos():
    records = [{"episode_index": i, "duplicate_id": i//2, "task": "pick" if i<1000 else "place", "speed_bin": 1}
               for i in range(2000)]
    splits = grouped_stratified_split(records)
    assert splits == grouped_stratified_split(records[::-1])
    assert {key: len(value) for key, value in splits.items()} == {"train": 1960, "validation": 20, "test": 20}
    membership = {i: name for name, ids in splits.items() for i in ids}
    assert all(membership[i] == membership[i+1] for i in range(0, 2000, 2))
    assert all(len([i for i in ids if i<1000]) == len(ids)//2 for ids in splits.values())


def pilots():
    return [{"variant": v, "lr": lr, "sampling": scheme, "complete": True, "initial_mse": 2., "final_mse": 1.}
            for v in ("dom100", "kr100") for lr in (1e-5, 3e-5, 1e-4) for scheme in ("frames", "blocks")]


def test_lr_selection_requires_both_models_and_block_speed_and_quality():
    rows = pilots()
    assert select_recipe(rows, 1.09)["sampling"] == "frames"
    assert select_recipe(rows, 1.10)["sampling"] == "blocks"
    assert select_recipe(rows, 1.10)["lr"] == 1e-5
    next(r for r in rows if r["variant"]=="kr100" and r["lr"]==3e-5 and r["sampling"]=="blocks")["final_mse"] = 1.06
    assert select_recipe(rows, 2.)["sampling"] == "frames"
    rows = [r for r in rows if not (r["variant"]=="dom100" and r["lr"]==1e-5 and r["sampling"]=="frames")]
    with pytest.raises(ValueError):
        select_recipe(rows, 1.)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.drop = torch.nn.Dropout(.2)
    def forward(self, x):
        return self.linear(self.drop(x))
    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True, parents=True)
        safetensors.torch.save_model(self, str(Path(path)/"model.safetensors"))
        (Path(path)/"config.json").write_text('{}')


@pytest.mark.parametrize("sampling", ["frames", "blocks"])
def test_real_optimizer_scheduler_rng_and_partial_epoch_resume(tmp_path, sampling):
    def setup():
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        model = Tiny(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = make_scheduler(optimizer, 6, warmup=2)
        return model, optimizer, scheduler
    def advance(model, optimizer, scheduler, progress, stop=None):
        while progress["epoch"] < 2:
            group = min(352, 725-progress["cursor"]); used=0
            for items in EpochBatchSampler([301, 424], 44, progress["epoch"], progress["cursor"], sampling=sampling):
                x = torch.tensor([[idx/725, idx%3, 1.] for _,idx in items])
                loss = model(x).square().mean()
                (loss*len(items)/group).backward()
                used += len(items); progress["cursor"] += len(items); progress["samples"] += len(items)
                if used != group:
                    continue
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                progress["updates"] += 1
                if progress["updates"] == stop:
                    return
                group = min(352, 725-progress["cursor"]); used=0
            progress["epoch"] += 1; progress["cursor"] = 0
    empty = lambda: {"epoch": 0, "cursor": 0, "samples": 0, "updates": 0}
    uninterrupted = setup(); all_progress = empty()
    advance(*uninterrupted, all_progress)
    interrupted = setup(); partial = empty()
    advance(*interrupted, partial, stop=2)
    assert partial["cursor"] == 704
    save_generation(tmp_path, *interrupted, partial, "frozen", "wandb42")
    restored = setup()
    state = load_generation(tmp_path, *restored, "frozen")
    assert state["wandb_id"] == "wandb42"
    advance(*restored, state["progress"])
    assert state["progress"] == all_progress == {"epoch": 2, "cursor": 0, "samples": 1450, "updates": 6}
    for key, value in uninterrupted[0].state_dict().items():
        torch.testing.assert_close(value, restored[0].state_dict()[key], rtol=0, atol=0)
    assert uninterrupted[2].state_dict() == restored[2].state_dict()
    for param in uninterrupted[1].state_dict()["state"]:
        for key, value in uninterrupted[1].state_dict()["state"][param].items():
            torch.testing.assert_close(value, restored[1].state_dict()["state"][param][key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_generation(tmp_path, *restored, "changed")


def test_checkpoint_pointers_are_atomic_and_writer_is_locked(tmp_path):
    model=Tiny(); optimizer=torch.optim.AdamW(model.parameters()); scheduler=make_scheduler(optimizer, 10)
    progress={"epoch": 1, "cursor": 0, "samples": 352, "updates": 1}
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass
    path=save_generation(tmp_path, model, optimizer, scheduler, progress, "f", "w", {"updates": 1, "value": 2}, True)
    orphan=tmp_path/"generations/.tmp-failed"; orphan.mkdir()
    (orphan/"model.safetensors").write_text("partial")
    assert resolve_generation(tmp_path) == resolve_generation(tmp_path, "best.json") == path
    next_progress={**progress, "updates": 2}
    save_generation(tmp_path, model, optimizer, scheduler, next_progress, "f", "w")
    assert path.exists() and resolve_generation(tmp_path) != path


def test_actual_policy_loss_uses_real_channels_padding_and_anchor_weights():
    source=Path(__file__).resolve().parents[1]/"policies/dynamicvla/modeling_dynamicvla.py"
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name=="DynamicVLAPolicy")
    node=next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=="forward")
    ns={"torch": torch, "OBS_STATE": "observation.state", "ACTION": "action"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    losses=torch.ones(2, 50, 32, requires_grad=True)
    with torch.no_grad():
        losses[0, :10, :7] = 2
        losses[0, 10:, :7] = 999
        losses[..., 7:] = 999
    mask=torch.zeros(2, 50, dtype=torch.bool); mask[0, 10:]=True
    fake=SimpleNamespace(config=SimpleNamespace(adapt_to_pi_aloha=False, enable_residual_flow=False,
        action_feature=SimpleNamespace(shape=(7,)), residual_position_key="pos", residual_lookahead_position_key="future",
        residual_lookahead_horizon_key="horizon"), extract_residual_batch_fields=lambda b: {},
        normalize_inputs=lambda b: b, normalize_targets=lambda b: b, prepare_images=lambda b: (None,None),
        prepare_state=lambda b: b["observation.state"], prepare_language=lambda b: (None,None),
        prepare_action=lambda b: b["action"], prepare_residual_xyz=lambda *a: None,
        prepare_residual_velocity=lambda *a: None, prepare_residual_horizon=lambda *a: None,
        get_action_normalization_std=lambda *a: None, prepare_phase_gate=lambda *a: None,
        model=SimpleNamespace(forward=lambda *a,**kw: losses))
    loss,_=ns["forward"](fake, {"observation.state": torch.zeros(2, 2, 6), "action": torch.zeros(2, 50, 7), "action_is_pad": mask})
    assert loss.item() == 1.5
    loss.backward()
    assert losses.grad[..., 7:].count_nonzero()==0
    assert losses.grad[0, 10:].count_nonzero()==0
    torch.testing.assert_close(losses.grad[0, :10, :7].sum(), losses.grad[1, :, :7].sum())


def test_six_independent_job_files_and_exact_architecture_flags():
    project=Path(__file__).resolve().parents[1]
    assert list(VARIANTS)==["dom100", "kr100", "domv100", "krv100", "krp100", "krvp100"]
    assert len(set(VARIANTS.values()))==6
    for i, variant in enumerate(VARIANTS,1):
        content=(project/"slurm"/f"{i:02d}_{variant}.sbatch").read_text()
        assert "--dependency" not in content
        assert "--gpus=a100:1" in content and "--time=48:00:00" in content
        assert f"train {variant}" in content


def test_diverged_lr_is_discarded_without_discarding_other_completed_pairs():
    rows=pilots()
    for row in rows:
        if row["lr"] == 1e-5:
            row["stable"]=False
    assert select_recipe(rows,1.2)["lr"] == 3e-5


def test_baseline_and_coordinate_read_checkpoint_normalizer_parameters():
    source=Path(__file__).resolve().parents[1]/"policies/dynamicvla/modeling_dynamicvla.py"
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="DynamicVLAPolicy")
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="get_action_normalization_std")
    ns={"torch":torch,"ACTION":"action"}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    normalizer=torch.nn.Module();normalizer.register_parameter("buffer_action_std",torch.nn.Parameter(torch.arange(1.,8.),requires_grad=False))
    for enabled in (False, True):
        policy=SimpleNamespace(config=SimpleNamespace(enable_residual_flow=enabled,residual_normalize_drift=True),
            normalize_targets=normalizer,unnormalize_outputs=normalizer)
        torch.testing.assert_close(ns['get_action_normalization_std'](policy,torch.device('cpu'),torch.float32),torch.arange(1.,8.))


@pytest.mark.parametrize("horizon", [20, 50])
def test_actual_live_gate_matches_offline_prefix_and_requires_applied_command(horizon):
    from proactive_vla.phase_gate import CausalPhaseGate,PhaseGateConfig,gate_from_state,replay_prefix_states
    source=Path(__file__).resolve().parents[1]/"scripts/inference.py"
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('_update_live_phase_gate','_get_environment_state_vector')]
    ns=dict(np=np,CausalPhaseGate=CausalPhaseGate,PhaseGateConfig=PhaseGateConfig,gate_from_state=gate_from_state)
    for node in nodes:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    policy=SimpleNamespace(config=SimpleNamespace(residual_phase_gate_config={},chunk_size=horizon,residual_action_dt=.04,residual_phase_gate_key='dynamic_object.phase_gate'))
    n=16;states=np.zeros((n,6));states[:,2]=np.arange(n)*.004
    env=np.zeros((n,9));env[:,2]=states[:,2]+.1;env[:,8]=.1
    actions=np.zeros((n,7));actions[:,-1]=-1;times=np.arange(n)*.04
    compact,_=replay_prefix_states(states,env,actions,times)
    for i in range(n):
        observation={'timestamp':times[i],'previous_gripper_command':None if i==0 else np.array([-1]),
            'observation.state':{'end_effector':{'pos':states[i:i+1,:3]}},'observation.environment_state':env[i:i+1]}
        ns['_update_live_phase_gate'](policy,observation)
        np.testing.assert_array_equal(observation['dynamic_object.phase_gate'][0],gate_from_state(compact[i],horizon,.04))
    del observation['previous_gripper_command']
    with pytest.raises(ValueError):ns['_update_live_phase_gate'](policy,observation)


def test_manipulation_results_use_termination_success_and_real_drops_collisions():
    from utils.dom_sim_metrics import ManipulationMetrics,summarize_cases
    metric=ManipulationMetrics()
    for i in range(8):
        z=i*.004
        metric.observe(i*.04,[0,0,z],[0,0,z+.1],[0,0,.1],-1,2 if i in (1,2,5) else 0)
    result=metric.finish({'object_dropping':True,'time_out':False},'place',False,True)
    assert result['grasp'] and result['lift']
    assert not result['completion'] and not result['place']
    assert result['drops']==1 and result['collisions']==2 and result['timeouts']==0
    assert summarize_cases([result])['completion']==0
    result['collisions']=None
    with pytest.raises(ValueError):summarize_cases([result])


def test_simulator_binding_matches_existing_nested_object_loader(tmp_path):
    from scripts.dom_closed_loop import scene_asset_paths
    rigid='isaaclab.assets.rigid_object.rigid_object:RigidObject'
    scene={
        'house': {'spawn': {'usd_path':'/original/scenes/room.usd'}},
        'object': {'class_type':rigid,'spawn':{'usd_path':'/original/objects/apple/model.usd'}},
        'object_2': {'class_type':rigid,'spawn':{'usd_path':'/original/objects/pear/model.usd'}},
        'robot': {'class_type':'Articulation','spawn':{'usd_path':'https://example/panda.usd'}}}
    assert scene_asset_paths(tmp_path,scene)==[
        tmp_path/'scenes/room.usd',tmp_path/'objects/apple/model.usd',tmp_path/'objects/pear/model.usd']


def test_prefetch_checks_pinned_git_and_lfs_without_network(tmp_path):
    import hashlib
    from scripts.prepare_dom100 import repo_file_matches
    path=tmp_path/'cached';path.write_bytes(b'cached demonstration')
    contents=path.read_bytes()
    lfs=SimpleNamespace(size=len(contents),lfs={'sha256':hashlib.sha256(contents).hexdigest()})
    git=SimpleNamespace(size=len(contents),lfs=None,
        blob_id=hashlib.sha1(f'blob {len(contents)}\0'.encode()+contents).hexdigest())
    assert repo_file_matches(path,lfs) and repo_file_matches(path,git)
    path.write_bytes(b'corrupt '+contents[8:])
    assert not repo_file_matches(path,lfs) and not repo_file_matches(path,git)
    assert not repo_file_matches(tmp_path/'absent',lfs)


def test_deferred_simulator_plan_is_frozen_in_recipe(tmp_path):
    from utils.dom_experiment import atomic_json,file_digest,simulator_evaluation_mode
    assert simulator_evaluation_mode(tmp_path)=='during_training'
    path=tmp_path/'evaluation_plan.json';atomic_json(path,{'mode':'after_training'})
    recipe={'simulator_evaluation_mode':'after_training','evaluation_plan_sha256':file_digest(path)}
    assert simulator_evaluation_mode(tmp_path,recipe)=='after_training'
    atomic_json(path,{'mode':'during_training'})
    with pytest.raises(ValueError):simulator_evaluation_mode(tmp_path,recipe)


def test_per_sample_augmentation_does_not_reseed_cuda(monkeypatch):
    seeded=[]
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:seeded.append(seed))
    original=torch.get_rng_state().clone()
    with sample_seed(123):first=torch.rand(4)
    with sample_seed(123):second=torch.rand(4)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert not seeded
    assert torch.equal(original,torch.get_rng_state())


class SpawnEpisodeBase:
    def __init__(self):
        self.episodes=[0]
        self.meta=SimpleNamespace(fps=25)
        self.episode_data_index={'length':torch.tensor([2])}
    def _get_episode(self,episode):
        return {'action':torch.full((2,7),float(episode))}


def test_worker_spawn_preserves_original_episode_reader():
    import pickle
    from utils.dom_data import DOMData
    data=pickle.loads(pickle.dumps(DOMData(SpawnEpisodeBase())))
    assert data._episode(0)['action'].shape==(2,7)
    torch.testing.assert_close(data.base._get_episode(0)['action'],torch.zeros(2,7))


class SpawnReaderData(DOMData):
    def __len__(self):return 2
    def __getitem__(self,index):
        if isinstance(index,tuple):index=index[1]
        return self._episode(0)['action'][index]
    def __getitems__(self,indices):return [self[index] for index in indices]


def test_real_loader_spawn_decodes_through_original_reader():
    data=SpawnReaderData(SpawnEpisodeBase())
    loader=torch.utils.data.DataLoader(data,batch_size=2,num_workers=2,multiprocessing_context='spawn')
    torch.testing.assert_close(next(iter(loader)),torch.zeros(2,7))


def test_lengths_own_storage_independently_of_worker_shared_tensors():
    base=SpawnEpisodeBase()
    data=DOMData(base)
    sampler=EpochBatchSampler(data.lengths,2,sampling='blocks')
    assert not np.shares_memory(data.lengths,base.episode_data_index['length'].numpy())
    assert not np.shares_memory(sampler.lengths,data.lengths)
    base.episode_data_index['length'].share_memory_()
    base.episode_data_index['length'].fill_(999999)
    data.lengths[0]=1000000
    assert sampler.lengths.tolist()==[2]
    assert sorted(i for batch in sampler for _,i in batch)==[0,1]


@pytest.mark.parametrize('sampling',['frames','blocks'])
def test_real_spawn_loader_preserves_sampler_lengths(sampling):
    data=SpawnReaderData(SpawnEpisodeBase())
    sampler=EpochBatchSampler(data.lengths,2,sampling=sampling)
    loader=torch.utils.data.DataLoader(data,batch_sampler=sampler,num_workers=2,multiprocessing_context='spawn')
    torch.testing.assert_close(next(iter(loader)),torch.zeros(2,7))
    assert data.lengths.tolist()==sampler.lengths.tolist()==[2]
