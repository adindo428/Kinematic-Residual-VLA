"""CPU reproduction using the actual frozen training and VLASH inference methods.

No weights or simulator are loaded. A recording tokenizer exposes the strings
each real method supplies, avoiding any dependence on tokenizer decoding.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'scratch/dom45/evaluations/multivla20k-20261029'
OUT = ROOT/'scratch/dom45/audits/kr-low-clutter-20261111'


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, **kwargs):
        self.calls.append({'prompts': prompts, 'kwargs': kwargs})
        return SimpleNamespace(input_ids=torch.zeros(1, 4, dtype=torch.long),
                               attention_mask=torch.ones(1, 4, dtype=torch.long))


class NativeTokenizer(Tokenizer):
    def __call__(self, prompts, **kwargs):
        value = super().__call__(prompts, **kwargs)
        return vars(value)


def method(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    wrapper = CODE/'model_code/vlash_kr/policy_inference.py'
    native = CODE/'vlash_source/vlash/policies/pi05/modeling_pi05.py'
    tokenizer = Tokenizer()
    fake = SimpleNamespace(velocity_mean=torch.zeros(3), recipe={'kr': False}, method='vlash',
        affine=lambda x, key: x, tokenizer=tokenizer,
        policy=SimpleNamespace(config=SimpleNamespace(tokenizer_max_length=200)))
    batch = {'observation.state': torch.zeros(1, 6), 'velocity': torch.zeros(1, 3),
             'action': torch.zeros(1, 20, 7), 'action_is_pad': torch.zeros(1, 20, dtype=torch.bool),
             'task': ['Place the cup on the plate.']}
    common_spec = importlib.util.spec_from_file_location('prompt_probe_common', wrapper.parent/'common.py')
    common = importlib.util.module_from_spec(common_spec); common_spec.loader.exec_module(common)
    prepare = method(wrapper, 'prepare', {'torch': torch, 'coordinates': common.coordinates, 'drift': common.drift})
    prepared, _ = prepare(fake, batch)
    native_tokenizer = NativeTokenizer()
    native_policy = SimpleNamespace(config=SimpleNamespace(state_cond=False, tokenizer_max_length=200),
                                    language_tokenizer=native_tokenizer)
    native_prepare = method(native, 'prepare_language', {'torch': torch, 'Tensor': torch.Tensor,
                                                        'OBS_STATE': 'observation.state'})
    native_prepare(native_policy, prepared, pad_to_max_length=False)
    training = tokenizer.calls[0]['prompts'][0]
    inference = native_tokenizer.calls[0]['prompts'][0]
    assert training != inference
    train_state = training.split('State: ')[1].split(';')[0].split()
    infer_state = inference.split('State: ')[1].split(';')[0].split()
    assert len(train_state) == 32 and len(infer_state) == 6
    assert set(train_state) == {'128'} and set(infer_state) == {'127'}
    # A repair can pass the already prepared tokens directly to sample_actions,
    # using the same image/state preparation as training. Do not mutate frozen suites.
    result = {'confirmed': True, 'training_prompt': training, 'actual_prediction_prompt': inference,
              'training_state_values': len(train_state), 'prediction_state_values': len(infer_state),
              'zero_state_training_bin': 128, 'zero_state_prediction_bin': 127,
              'affected_models': ['vlash_baseline', 'vlash_kr'],
              'root_cause': 'Wrapper.forward uses its prepared language tokens; wrapper.predict calls VLASH.predict_action_chunk, which ignores them and retokenizes batch task/state.',
              'same_issue_in_pi05': False,
              'same_issue_in_smolvla': False,
              'score_or_checkpoint_modified': False}
    (OUT/'vlash_prompt_reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
