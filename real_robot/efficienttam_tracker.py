"""Causal, bounded streaming adapter for the official EfficientTAM predictor.

Uses public init/prompt/propagation methods from yformer/EfficientTAM at
abcd061ebd3cc6e7527d152d75b890126aaa53f6 (Apache-2.0). One model is shared;
each camera has its own inference state and manually selected object prompt.
No color thresholding is used to propagate masks. No actuator imports.
"""
from contextlib import nullcontext
from functools import partial
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from .geometry import transform_point
from .perception import depth_measurement


class EfficientTAMTracker:
    def __init__(self, checkpoint, output, config='configs/efficienttam/efficienttam_ti_512x512.yaml', device='cuda'):
        import torch
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        import efficient_track_anything.efficienttam_video_predictor as predictor_module
        # Suppress one progress bar per live frame, without hiding warnings.
        predictor_module.tqdm = partial(predictor_module.tqdm, disable=True)
        self.torch = torch; self.device = device
        torch.set_num_threads(4)
        self.predictor = build_efficienttam_video_predictor(config, str(checkpoint), device=device,
            apply_postprocessing=False, hydra_overrides_extra=['++model.compile_image_encoder=false', '++model.fill_hole_area=0'])
        self.output = Path(output); self.states = {}; self.indices = {}; self.stamps = {}
        self.seed_boxes = {}; self.seed_depths = {}; self.last_recovery = {}
        self.provenance = {'tracker': 'EfficientTAM', 'upstream': 'https://github.com/yformer/EfficientTAM',
            'upstream_commit': 'abcd061ebd3cc6e7527d152d75b890126aaa53f6', 'config': config,
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'torch_version': torch.__version__, 'device': device, 'retained_nonconditioning_frames': 32,
            'online_forward_only': True, 'model_shared_between_views': True,
            'inference_states_shared_between_views': False, 'color_threshold_mask_propagation': False,
            'recovery': 'Other-view 3D guided box re-prompt, at most 1 Hz. Require >=30% blue pixels and <=60 mm cross-view depth agreement; rejected proposals leave model memory unchanged.'}

    def _context(self):
        return self.torch.autocast('cuda', dtype=self.torch.bfloat16) if self.device.startswith('cuda') else nullcontext()

    def _tensor(self, bgr):
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        array = np.array(Image.fromarray(rgb).resize((self.predictor.image_size, self.predictor.image_size)))
        tensor = self.torch.from_numpy(array.copy()).permute(2, 0, 1).float().to(self.device)/255.
        mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
        std = tensor.new_tensor([.229, .224, .225])[:, None, None]
        return (tensor-mean)/std

    def seed(self, role, bgr, stamp, box):
        x, y, w, h = map(int, box); height, width = bgr.shape[:2]
        if x < 0 or y < 0 or min(w, h) < 5 or x+w > width or y+h > height:
            raise ValueError('Select a nonempty box around the tape within the image')
        folder = self.output/('prompt_'+role); folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder/'00000.jpg'), bgr)
        with self.torch.inference_mode(), self._context():
            state = self.predictor.init_state(str(folder))
            # Keep upstream initialization, but infer from the uncompressed live
            # frame and retain only the current image tensor from now onward.
            state['images'] = {0: self._tensor(bgr)}; state['cached_features'].clear()
            self.predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.array([x, y, x+w, y+h], np.float32))
            _, _, logits = next(self.predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0))
            self.states[role] = state; self.indices[role] = 0; self.stamps[role] = stamp
            mask, info = self._mask(role, logits)
        info.update(prompt_box_xywh=[x, y, w, h], seed=True)
        self.seed_boxes[role] = [x, y, w, h]
        return mask, info

    def track(self, role, bgr, stamp):
        state = self.states[role]
        if stamp <= self.stamps[role]: raise ValueError('EfficientTAM frames must advance in capture time')
        if bgr.shape[:2] != (state['video_height'], state['video_width']):
            raise ValueError('Camera resolution changed after prompting')
        before = time.monotonic(); gap = stamp-self.stamps[role]
        self.stamps[role] = stamp; self.indices[role] += 1; index = self.indices[role]
        with self.torch.inference_mode(), self._context():
            state['images'] = {index: self._tensor(bgr)}; state['num_frames'] = index+1
            _, _, logits = next(self.predictor.propagate_in_video(state, start_frame_idx=index, max_frame_num_to_track=0))
            mask, info = self._mask(role, logits)
        prune_state(state, index)
        info.update(inference_s=time.monotonic()-before, processed_capture_gap_s=gap, seed=False)
        return mask, info

    def recover_pair(self, frames, candidates, infos, T_front_side):
        """Use the still-tracked view to propose, then verify a lost-view mask."""
        fs = np.asarray(T_front_side); inverse = np.linalg.inv(fs)
        for role in ('front', 'side'):
            valid = [c for c in candidates[role] if c['point'] is not None]
            if valid:
                self.seed_depths.setdefault(role, float(valid[0]['point'][2]))
                continue
            other = 'side' if role == 'front' else 'front'
            other_valid = [c for c in candidates[other] if c['point'] is not None]
            stamp = frames[role]['timing']['sensor_timestamp_s']
            if (len(other_valid) != 1 or infos[other]['object_probability'] < .9
                    or role not in self.seed_depths or self.indices[role] == 0
                    or stamp-self.last_recovery.get(role, -float('inf')) < 1.):
                continue
            self.last_recovery[role] = stamp
            other_point = other_valid[0]['point']
            predicted = transform_point(fs if role == 'front' else inverse, other_point)
            pixel = project_native_pixel(predicted, frames[role])
            if pixel is None: continue
            x, y = pixel; old_box = self.seed_boxes[role]
            scale = self.seed_depths[role]/predicted[2]
            width, height = frames[role]['bgr'].shape[1], frames[role]['bgr'].shape[0]
            half_w, half_h = old_box[2]*scale/2+20, old_box[3]*scale/2+20
            left, top = max(0, int(x-half_w)), max(0, int(y-half_h))
            right, bottom = min(width, int(x+half_w+1)), min(height, int(y+half_h+1))
            box = [left, top, right-left, bottom-top]
            mask, info, proposal = self._propose(role, box)
            accepted = False; distance = blue_fraction = None
            if mask is not None:
                hsv = cv2.cvtColor(frames[role]['bgr'], cv2.COLOR_BGR2HSV)
                blue = cv2.inRange(hsv, np.array([90, 60, 25], np.uint8), np.array([135, 255, 255], np.uint8))
                blue_fraction = float(np.mean(blue[mask > 0] > 0))
                point, quality = depth_measurement(mask, frames[role]['depth'], frames[role]['deproject'],
                                                   .1 if role == 'front' else .07, 1.5 if role == 'front' else .5)
                if point is not None:
                    distance = float(np.linalg.norm(point-predicted))
                    accepted = blue_fraction >= .3 and distance <= .06
                if accepted:
                    self.states[role] = proposal
                    candidates[role] = [{'point': point, 'quality': quality, 'mask': mask, 'candidate_index': 0}]
                    infos[role] = {**infos[role], **info}
            infos[role]['auto_recovery'] = {'accepted': accepted, 'box_xywh': box,
                'blue_fraction': blue_fraction, 'cross_view_distance_m': distance}

    def _propose(self, role, box):
        original = self.states[role]; trial = copy_buffers(original); index = self.indices[role]
        # The original prompt stays available, together with one latest recovery
        # prompt. Dictionary copies isolate the trial; tensor buffers are read-only.
        for outputs in trial['output_dict_per_obj'].values():
            outputs['non_cond_frame_outputs'].clear()
            for old in list(outputs['cond_frame_outputs']):
                if old != 0: del outputs['cond_frame_outputs'][old]
        for key in ('point_inputs_per_obj', 'mask_inputs_per_obj'):
            for prompts in trial[key].values():
                for old in list(prompts):
                    if old != 0: del prompts[old]
        trial['frames_tracked_per_obj'][0].pop(index, None)
        self.states[role] = trial
        try:
            x, y, w, h = box
            with self.torch.inference_mode(), self._context():
                self.predictor.add_new_points_or_box(trial, frame_idx=index, obj_id=1,
                    box=np.array([x, y, x+w, y+h], np.float32))
                _, _, logits = next(self.predictor.propagate_in_video(trial, start_frame_idx=index, max_frame_num_to_track=0))
                mask, info = self._mask(role, logits)
            info['seed'] = False; info['recovered'] = True
            return mask, info, trial
        finally:
            self.states[role] = original

    def _mask(self, role, logits):
        index = self.indices[role]; state = self.states[role]
        outputs = state['output_dict_per_obj'][0]
        out = outputs['cond_frame_outputs'].get(index, outputs['non_cond_frame_outputs'].get(index))
        score = float(out['object_score_logits'].float().sigmoid().item())
        mask = (logits[0, 0] > 0).to('cpu').numpy().astype(np.uint8)*255
        # Preserve model holes. Erode only the boundary used for metric depth.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        area = int(np.count_nonzero(mask))
        reason = 'model_object_not_visible' if score < .5 else ('model_mask_too_small' if area < 80 else None)
        return (None if reason else mask), {'tracker': 'EfficientTAM', 'frame_index': index,
            'object_probability': score, 'mask_pixels': area, 'mask_reason': reason}


def prune_state(state, index, retain=32):
    """Keep the original prompt and sufficient causal mask/pointer memory."""
    for outputs in state['output_dict_per_obj'].values():
        history = outputs['non_cond_frame_outputs']
        for old in list(history):
            if old <= index-retain: del history[old]
    for tracked in state['frames_tracked_per_obj'].values():
        for old in list(tracked):
            if old != 0 and old <= index-retain: del tracked[old]


def copy_buffers(value):
    """Copy mutable containers, sharing read-only inference tensor buffers."""
    if isinstance(value, dict): return type(value)((k, copy_buffers(v)) for k, v in value.items())
    if isinstance(value, list): return [copy_buffers(v) for v in value]
    if isinstance(value, tuple): return tuple(copy_buffers(v) for v in value)
    return value


def project_native_pixel(point, frame):
    """Invert the camera's native deprojection, including inverse-Brown lenses."""
    from scipy.optimize import least_squares
    p = np.asarray(point, float); intr = frame['timing']['intrinsics']
    if p.shape != (3,) or not np.isfinite(p).all() or p[2] <= .03: return None
    target = p[:2]/p[2]
    initial = target*[intr['fx'], intr['fy']]+[intr['ppx'], intr['ppy']]
    upper = np.array([intr['width']-1., intr['height']-1.])
    def residual(pixel):
        ray = np.array(frame['deproject'](pixel.tolist(), 1.))
        return ray[:2]/ray[2]-target
    result = least_squares(residual, np.clip(initial, 0, upper), bounds=(np.zeros(2), upper),
                           diff_step=1e-3, max_nfev=30, gtol=1e-10, ftol=1e-10, xtol=1e-10)
    return result.x if result.success and np.linalg.norm(result.fun) < 1e-5 else None
