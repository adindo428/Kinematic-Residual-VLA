"""Reusable, camera-specific EfficientTAM reference images and masks."""
import json
from pathlib import Path
import shutil

import cv2


def load_prompt(root, role, serial):
    root = Path(root)
    meta = json.loads((root / (role+'-prompt.json')).read_text())
    if meta['timing']['camera_serial'] != serial:
        raise ValueError(f'{role} saved mask belongs to another camera; create a new object mask')
    folder = root / ('prompt_'+role)
    image = cv2.imread(str(folder / 'reference.png')) if (folder/'reference.png').exists() else None
    if image is None:
        image = cv2.imread(str(folder / '00000.jpg'))
    mask_path = folder / 'mask.png'
    mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if image is None or (mask is not None and mask.shape != image.shape[:2]):
        raise ValueError(f'Invalid saved {role} reference image/mask')
    return image, mask, meta


def save_prompt(source, destination, role):
    source, destination = Path(source), Path(destination)
    folder = destination / ('prompt_'+role)
    folder.mkdir(parents=True,exist_ok=True)
    # Publish metadata last so readers never see a complete role without its assets.
    for name in ('00000.jpg','reference.png','mask.png'):
        src=source/('prompt_'+role)/name
        if src.exists():
            tmp=folder/(name+'.tmp')
            shutil.copyfile(src,tmp)
            tmp.replace(folder/name)
    meta=destination/(role+'-prompt.json')
    tmp=meta.with_suffix('.tmp')
    shutil.copyfile(source/meta.name,tmp)
    tmp.replace(meta)


def has_prompt(root, role):
    return (Path(root)/(role+'-prompt.json')).is_file()


def find_tape_reference(project):
    """Only import sessions explicitly identified as blue tape by saved grasps."""
    candidates=[]
    for path in (Path(project)/'scratch/real_robot/grasps').glob('*/manifest.json'):
        meta=json.loads(path.read_text())
        if meta.get('object') != 'blue_tape' or not meta.get('observation_file'):
            continue
        root=Path(meta['observation_file']).parent
        if all(has_prompt(root,role) for role in ('front','side')):
            candidates.append((meta.get('created_utc',''),root))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None
