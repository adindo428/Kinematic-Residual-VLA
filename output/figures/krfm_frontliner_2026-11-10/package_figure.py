"""Embed original PNG bytes in the SVG and record source provenance; no pixel editing."""
from pathlib import Path
import base64
import hashlib
import json
import re

root = Path(__file__).resolve().parent
svg = root / 'krfm_frontliner_publication_v1.svg'
text = svg.read_text()
text = re.sub(r'xlink:href="(sources/[^\"]+\.png)"', lambda m: 'xlink:href="data:image/png;base64,' + base64.b64encode((root / m.group(1)).read_bytes()).decode() + '"', text)
svg.write_text(text)
source = 'https://huggingface.co/datasets/ailab-Clemson/KRVLA-dataset/resolve/44359e9169c5d0f5460f78f93e09c20c96d7d96b/videos/chunk-000/observation.images.opst_cam/episode_000285.mp4'
manifest = {
 'created': '2026-11-10',
 'real': {'source_url': source, 'episode': 285, 'kind': 'human leader/follower demonstration', 'fps': 25, 'frame_indices': [25,40,55], 'timestamps_s': [1.0,1.6,2.2], 'crop_xywh': [0,96,384,288], 'crop_reason': 'Remove only black padding at top of video'},
 'simulation': {'source': 'docs/profiles/KRVLA_Private_Visuals_2026-11-07/videos/isaac_sim_krv_policy_success.mp4', 'original_source': 'scratch/dom45/evaluations/krv-full-original-dom-20261106/passes/01/full/00_CR/videos/krv100_full/0000/1-1_place_franka_apple13d_O02_01048988_09b0-1106-141856-SUCCESS.mp4', 'kind': 'learned-policy evaluation rollout', 'checkpoint': 'KRV100_Full validation-selected epoch 3; constant gate', 'fps':24, 'selected_frame_indices': [10,24,39], 'timestamps_s': [10/24,1.0,39/24], 'crop_xywh': [480,0,480,360], 'crop_reason': 'Extract fixed opposite camera from three-camera video'},
 'publication_assets': {'photographs': 'Original extracted PNG bytes embedded unchanged in SVG; displayed with scaling only.', 'annotations': 'Manual qualitative image-plane illustration, not calibrated trajectories or optical-flow estimation.', 'action_space_diagram': 'Schematic normalized action-space decomposition, not a projection measured in the images.', 'real_policy_claim': 'No real-world KRFM policy performance is claimed.'},
 'generated_concept': {'file':'krfm_frontliner_concept_v1.png','tool':'built-in image_gen','prompt':'generation_prompt_used.txt','use':'Design exploration. May contain generatively altered scene details; use original-frame publication assets for scientific evidence.'},
 'design_references': ['https://tracevla.github.io/', 'https://arxiv.org/html/2601.22153v1', 'https://openvla.github.io/'],
 'files_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'sources').glob('*')) if p.is_file()}
}
(root/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Embedded original frames and wrote provenance.json')
