"""KRFM wrapper around pinned official LeRobot and VLASH action experts.

Residual targets are formed in meters, before the native normalization. The
same affine transform is inverted at inference and physical drift restored
exactly once. This does not add object velocity to the neural prefix.
"""
import json
from pathlib import Path

import torch
from torch import nn
from common import coordinates, drift, masked_loss, sha


class Policy(nn.Module):
    def __init__(self, root, recipe, device):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        root = Path(root)
        self.recipe = recipe
        self.method = recipe['method']
        self.stats = json.loads((root / 'statistics.json').read_text())
        data = json.loads((root / 'data-provenance.json').read_text())
        self.register_buffer('velocity_mean', torch.tensor(data['velocity_mean']))
        refs = json.loads((root / 'checkpoints.json').read_text())
        model_name = 'lerobot/smolvla_base' if self.method == 'smolvla' else 'lerobot/pi05_base'
        checkpoint = Path(refs[model_name]['path'])
        if sha(checkpoint / 'model.safetensors') != refs[model_name]['files']['model.safetensors']:
            raise ValueError('Pretrained checkpoint bytes changed')
        if self.method == 'smolvla':
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as Config
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Model
        elif self.method == 'pi05':
            from lerobot.policies.pi05.configuration_pi05 import PI05Config as Config
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Model
        else:
            from vlash_runtime import activate
            activate()
            from vlash.policies.pi05.configuration_pi05 import PI05Config as Config
            from vlash.policies.pi05.modeling_pi05 import PI05Policy as Model
        config = Config()
        config.device = device
        config.chunk_size = config.n_action_steps = 20
        config.input_features = {
            'observation.images.opst_cam': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 480)),
            'observation.images.wrist_cam': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 480)),
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
        config.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        config.compile_model = False
        if self.method == 'vlash':
            config.state_cond = False  # retain original PI05 architecture; optional enhancement excluded
            config.normalization_mapping = {k: NormalizationMode.IDENTITY for k in ['VISUAL', 'STATE', 'ACTION']}
            config.fuse_qkv = config.fuse_gate_up = False
        elif self.method == 'pi05':
            config.gradient_checkpointing = True
        else:
            # Keep the original paper/official fine-tuning trainable set.
            config.freeze_vision_encoder = config.train_expert_only = True
            config.train_state_proj = True
            config.load_vlm_weights = False
            config.vlm_model_name = refs['HuggingFaceTB/SmolVLM2-500M-Video-Instruct']['path']
        self.policy = Model(config)
        weights = load_file(checkpoint / 'model.safetensors')
        if self.method == 'pi05':
            weights = self.policy._fix_pytorch_state_dict_keys(weights, config)
            weights = {k if k.startswith('model.') else 'model.' + k: v for k, v in weights.items()}
        elif self.method == 'vlash':
            rules = [('action_in_proj.', 'model.suffix_embedder.action_in_proj.'),
                     ('action_out_proj.', 'model.action_out_proj.'),
                     ('time_mlp_in.', 'model.suffix_embedder.time_mlp_in.'),
                     ('time_mlp_out.', 'model.suffix_embedder.time_mlp_out.'),
                     ('paligemma_with_expert.gemma_expert.', 'model.action_expert.'),
                     ('paligemma_with_expert.paligemma.', 'model.vlm.')]
            mapped = {}
            for key, val in weights.items():
                for old, new in rules:
                    if key.startswith(old):
                        key = new + key[len(old):]
                        break
                mapped[key] = val
            weights = mapped
        # Safetensors omits tied embedding aliases; VLASH additionally shares
        # layer modules through wrappers. Populate only aliases of the exact
        # same Parameter object, never randomly initialize missing weights.
        aliases = {}
        for name, param in self.policy.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(param), []).append(name)
        for names in aliases.values():
            present = [name for name in names if name in weights]
            if present:
                for name in names:
                    weights.setdefault(name, weights[present[0]])
        # Fail closed: upstream PI05 loading can otherwise return random weights
        # on a fetch error; VLASH's loader permits unexamined missing keys.
        self.policy.load_state_dict(weights, strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(refs[
            'HuggingFaceTB/SmolVLM2-500M-Video-Instruct' if self.method == 'smolvla'
            else 'google/paligemma-3b-pt-224']['path'])
        self.to(device)
        self._affine_buffers = {}
        for key in ('observation.state','action'):
            stats = self.stats[key]
            if self.method == 'smolvla':
                offset, scale = stats['mean'], stats['std']
            else:
                offset = [(a+b)/2 for a,b in zip(stats['q01'],stats['q99'])]
                scale = [max((b-a)/2,1e-6) for a,b in zip(stats['q01'],stats['q99'])]
            names = []
            for label, values in [('offset',offset),('scale',scale)]:
                name = '_affine_'+key.replace('.','_')+'_'+label
                self.register_buffer(name,torch.tensor(values,device=device),persistent=False)
                names.append(name)
            self._affine_buffers[key] = names

    def affine(self, x, key, inverse=False):
        offset, scale = [getattr(self,name).to(dtype=x.dtype) for name in self._affine_buffers[key]]
        return x * scale + offset if inverse else (x - offset) / scale

    def prepare(self, batch):
        device = self.velocity_mean.device
        b = {k: v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        correction = drift(b['velocity'], self.velocity_mean, 20, b['offset']) if self.recipe['kr'] else torch.zeros_like(b['action'])
        # Do not hand environment state / velocity / future labels to a policy.
        model_batch = {k: v for k, v in b.items() if k.startswith('observation.images.')}
        state = self.affine(b['observation.state'], 'observation.state')
        model_batch['observation.state'] = state
        model_batch['action'] = self.affine(coordinates(b['action'], correction), 'action')
        model_batch['action_is_pad'] = b['action_is_pad']
        tasks = b['task']
        if self.method != 'smolvla':
            padded = torch.nn.functional.pad(state, (0, 32-state.shape[-1]))
            bins = torch.linspace(-1, 1, 257, device=device)[:-1]
            values = (torch.bucketize(padded, bins, right=True) - 1).cpu().tolist()
            tasks = [f'Task: {t.strip().replace(chr(10), " ")}, State: {" ".join(map(str,s))};\nAction: ' for t,s in zip(tasks,values)]
        else:
            tasks = [t if t.endswith('\n') else t+'\n' for t in tasks]
        tokens = self.tokenizer(tasks, padding='max_length', truncation=True,
                                max_length=self.policy.config.tokenizer_max_length, return_tensors='pt')
        model_batch['observation.language.tokens'] = tokens.input_ids.to(device)
        model_batch['observation.language.attention_mask'] = tokens.attention_mask.to(device).bool()
        model_batch['task'] = b['task']
        return model_batch, correction

    def forward(self, batch, noise=None, time=None):
        b, _ = self.prepare(batch)
        p = self.policy
        tokens, masks = b['observation.language.tokens'], b['observation.language.attention_mask']
        actions = p.prepare_action(b)
        if self.method == 'pi05':
            images, image_masks = p._preprocess_images(b)
            losses = p.model.forward(images, image_masks, tokens, masks, actions, noise, time)
        else:
            images, image_masks = p.prepare_images(b)
            state = p.prepare_state(b)
            losses = p.model.forward(images, image_masks, tokens, masks, state, actions, noise, time)
        # Retain each pinned implementation's native loss-channel convention:
        # PI05 slices to real action features; SmolVLA/VLASH train padded32.
        return masked_loss(losses, b['action_is_pad'], 7 if self.method=='pi05' else 32)

    @torch.no_grad()
    def predict(self, batch):
        b, correction = self.prepare(batch)
        prediction = self.policy.predict_action_chunk(b)
        return coordinates(self.affine(prediction[..., :7], 'action', inverse=True), correction, inverse=True)
