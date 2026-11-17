"""Isolated, incremental EfficientTAM worker; protocol contains no object truth."""
import contextlib
import importlib
from pathlib import Path
import pickle
import sys


class StreamingMasks:
    def __init__(self, source, checkpoint, device='cuda'):
        sys.path.insert(0, str(source))
        import torch
        torch.set_num_threads(2)
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        self.torch = torch
        self.predictor = build_efficienttam_video_predictor(
            'configs/efficienttam/efficienttam_ti_512x512.yaml', str(checkpoint), device=device,
            vos_optimized=False, apply_postprocessing=False,
            hydra_overrides_extra=['++model.compile_image_encoder=false'])
        self.module = importlib.import_module(type(self.predictor).__module__)
        self.state = None
        self.index = -1
        self.device = device

    def reset(self):
        self.state = None
        self.index = -1

    def step(self, rgb, initial_mask=None):
        import numpy as np
        from PIL import Image
        torch = self.torch
        self.index += 1
        i = self.index
        h, w = rgb.shape[:2]
        image = np.asarray(Image.fromarray(rgb.astype('uint8')).resize(
            (self.predictor.image_size, self.predictor.image_size)))
        image = torch.as_tensor(image.copy(), device=self.device).permute(2, 0, 1).float()/255.
        mean = image.new_tensor([.485, .456, .406])[:, None, None]
        std = image.new_tensor([.229, .224, .225])[:, None, None]
        image = (image-mean)/std
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.startswith('cuda') else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            if self.state is None:
                assert i == 0 and initial_mask is not None
                loader = self.module.load_video_frames
                # Only the current frame is available. Never preload a future frame.
                self.module.load_video_frames = lambda **_: (image[None], h, w)
                try:
                    self.state = self.predictor.init_state('causal-live-frame', offload_state_to_cpu=True)
                finally:
                    self.module.load_video_frames = loader
                self.state['images'] = {i: image}
                self.predictor.add_new_mask(self.state, frame_idx=0, obj_id=1, mask=initial_mask)
            else:
                assert initial_mask is None, 'Only the first frame may receive a privileged mask'
                self.state['images'] = {i: image}
                self.state['num_frames'] = i+1
            result = list(self.predictor.propagate_in_video(
                self.state, start_frame_idx=i, max_frame_num_to_track=0))
            assert len(result) == 1 and result[0][0] == i
            mask = (result[0][2][0, 0] > 0).cpu().numpy()
            # Keep initial conditioning plus a conservative 64-frame temporal window.
            for out in self.state['output_dict_per_obj'].values():
                for key in list(out['non_cond_frame_outputs']):
                    if key <= i-64:
                        del out['non_cond_frame_outputs'][key]
            for tracked in self.state['frames_tracked_per_obj'].values():
                for key in list(tracked):
                    if key <= i-64:
                        del tracked[key]
            return mask


def main():
    request, response = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # Model diagnostics must not corrupt the binary protocol.
    tracker = StreamingMasks(Path(sys.argv[1]), Path(sys.argv[2]))
    pickle.dump({'ready': True}, response);response.flush()
    while True:
        try:
            data = pickle.load(request)
        except EOFError:
            return
        if data.get('reset'):
            tracker.reset();result = {'reset': True}
        else:
            result = {'mask': tracker.step(data['rgb'], data.get('initial_mask'))}
        pickle.dump(result, response, protocol=4);response.flush()


if __name__ == '__main__':
    main()
