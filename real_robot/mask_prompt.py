"""Interactive live-mask selection with retries and saved-profile backup."""
import json
from pathlib import Path
import time
import cv2
from .object_profiles import has_prompt,save_prompt


def status(root,state,role,reason=None):
    path=Path(root)/'prompt_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,role=role,reason=reason,host_monotonic_s=time.monotonic()),indent=2)+'\n')
    tmp.replace(path)


def seed_live(model,role,frame,root,profile=None,box=None,retry=False,automatic=False):
    root=Path(root);reason=None
    (root/'MASK_PENDING').touch()
    proposal=None;automatic_failure=None
    if automatic and box is None:
        try:
            if profile is None:raise ValueError('Automatic selection requires a saved object reference')
            from .automatic_mask import propose_box
            box,proposal=propose_box(frame['bgr'],profile,role,frame['timing']['camera_serial'])
            status(root,'automatic_selection',role,'Fresh live-frame proposal from saved object appearance')
        except (ValueError,OSError,KeyError) as exc:
            automatic_failure=str(exc);reason=automatic_failure
    while True:
        if box is None:
            title='Select object in '+role.upper()+' view, then press ENTER'
            status(root,'awaiting_selection',role,reason)
            print(title+((': '+reason) if reason else ''),flush=True)
            box=cv2.selectROI(title,frame['bgr'],showCrosshair=True,fromCenter=False)
            cv2.destroyWindow(title)
            if min(box[2:])<5:
                status(root,'cancelled',role,'Object selection cancelled')
                raise ValueError('Object selection cancelled; draw a nonempty box around the ball')
        mask,info=model.seed(role,frame['bgr'],frame['timing']['sensor_timestamp_s'],box)
        if mask is None:
            reason='No valid mask from this box; draw a tighter box around the visible ball'
            if not retry:
                status(root,'failed',role,reason)
                raise ValueError(reason+'; restart with --new-mask')
            if proposal is not None:automatic_failure='EfficientTAM rejected the automatic proposal'
            box=None;proposal=None
            continue
        info['selection_mode']='automatic' if proposal is not None else 'manual_or_explicit_box'
        if proposal is not None:info['automatic_proposal']=proposal
        if automatic_failure is not None:info['automatic_proposal_failure']=automatic_failure
        (root/(role+'-prompt.json')).write_text(json.dumps(dict(box_xywh=list(map(int,box)),timing=frame['timing'],model=info),indent=2)+'\n')
        if profile:
            backup=root/'previous_profile'
            if has_prompt(profile,role) and not has_prompt(backup,role):save_prompt(profile,backup,role)
            save_prompt(root,profile,role)
        status(root,'selected',role,'Mask selected; waiting for fresh depth and velocity tracking')
        (root/'MASK_PENDING').unlink(missing_ok=True)
        return mask,info
