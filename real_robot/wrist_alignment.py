"""Tape mask geometry and bounded eye-in-hand corrections. No hardware access."""
import cv2
import numpy as np


def gripper_gap(image, tape_mask=None):
    """Locate the two dark finger silhouettes connected to the image bottom."""
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    dark = (hsv[:,:,2]<85).astype(np.uint8)
    dark[:image.shape[0]//2] = 0
    # Dark fingers acquire blue hue under reflected illumination. Exclude the
    # actual TAM object mask rather than removing every blue-looking pixel.
    if tape_mask is not None:dark[tape_mask>0] = 0
    dark = cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers = cv2.connectedComponentsWithStats(dark)
    fingers = []
    for i in range(1,n):
        x,y,w,h,area = stats[i]
        if area<300 or y+h<image.shape[0]-2 or h<image.shape[0]*.15:continue
        ys,xs = np.where(labels==i)
        top = ys<=np.percentile(ys,2)
        fingers.append((area,np.array([np.median(xs[top]),np.median(ys[top])],float),centers[i,0]))
    left = [f for f in fingers if f[2]<image.shape[1]/2]
    right = [f for f in fingers if f[2]>=image.shape[1]/2]
    if not left or not right:return None
    a,b = max(left,key=lambda f:f[0])[1],max(right,key=lambda f:f[0])[1]
    span = float(np.linalg.norm(b-a))
    if span<80:return None
    return {'left_tip_px':a.tolist(),'right_tip_px':b.tolist(),
            'center_px':((a+b)/2).tolist(),'closing_axis_px':((b-a)/span).tolist(),'span_px':span}


def fixed_gripper_gap(reference, image_shape, serial):
    """Saved open-jaw geometry; scene content never changes the grasp center.

    Fingers move horizontally when opening; alignment uses their fixed midpoint.
    The displayed tips are the open reference, not estimated current positions.
    """
    if reference['camera_serial'] != serial or reference['image_size_wh'] != list(image_shape[1::-1]):
        raise ValueError('Reference camera/resolution changed; press G')
    tips = np.asarray(reference['tips_px'], dtype=float)
    if (tips.shape != (2,2) or not np.isfinite(tips).all()
            or tips[1,0] <= tips[0,0] or np.any(tips < 0)
            or np.any(tips >= np.array(image_shape[1::-1]))):
        raise ValueError('Invalid saved finger pixels; press G')
    a,b = tips
    return {'left_tip_px':a.tolist(), 'right_tip_px':b.tolist(),
            'center_px':((a+b)/2).tolist(), 'closing_axis_px':[1.,0.],
            'span_px':float(b[0]-a[0]), 'source':'saved_open_jaw_reference'}


def select_gripper_reference(image):
    """Explicit fingertip clicks, without silhouette/color detection."""
    title = 'Open gripper: LEFT then RIGHT inner tip; ENTER saves; ESC cancels'
    points = []
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) == 2: points.clear()
            points.append([x,y])
    cv2.namedWindow(title)
    cv2.setMouseCallback(title,click)
    try:
        while True:
            view = image.copy()
            for point in points:
                cv2.drawMarker(view,tuple(point),(255,0,255),cv2.MARKER_CROSS,16,2)
            if len(points) == 2: cv2.line(view,tuple(points[0]),tuple(points[1]),(255,0,255),2)
            cv2.imshow(title,view)
            key = cv2.waitKey(20)&255
            if key == 27 or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE) < 1:return None
            if key in (10,13) and len(points) == 2 and points[1][0] > points[0][0]:return points
    finally:
        cv2.destroyWindow(title)


def line_error(target, actual):
    """Undirected line angle error in radians, respecting parallel-jaw symmetry."""
    return float((target-actual+np.pi/2) % np.pi-np.pi/2)


def blue_seed(image, expected=(350, 260)):
    """Color is used only to propose an initial prompt for EfficientTAM.

    Saturation excludes the blue illumination reflected from the wooden table.
    The live identity/mask is subsequently supplied by EfficientTAM.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 150, 35]), np.array([130, 255, 230]))
    mask[:15] = 0
    mask[:, :35] = 0
    mask[:, -35:] = 0
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, n) if stats[i, 4] >= 120
                  and np.linalg.norm(centers[i]-expected) < 180]
    if not candidates:
        raise ValueError('No unambiguous blue tape prompt near the wrist working region')
    ranked = sorted(candidates, key=lambda i: np.linalg.norm(centers[i]-expected))
    if len(ranked) > 1 and abs(np.linalg.norm(centers[ranked[0]]-expected)-
                                np.linalg.norm(centers[ranked[1]]-expected)) < 35:
        raise ValueError('Multiple blue objects near wrist target; supply an explicit wrist prompt')
    i = ranked[0]
    x, y, w, h, _ = stats[i]
    pad = 8
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
    return [int(x0), int(y0), int(x1-x0), int(y1-y0)], (labels == i).astype(np.uint8)*255


def tape_feature(image, mask, depth=None):
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError('Tape mask unavailable')
    # Estimate the axis from saturated tape pixels inside the identity mask;
    # do not include the lamp reflection or dark gripper fingers in the PCA.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    core = ((mask > 0) & (hsv[:,:,0]>=90) & (hsv[:,:,0]<=140)
            & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 25))
    ys, xs = np.where(core)
    if len(xs) < 100:
        raise ValueError('Too few tape pixels for wrist geometry')
    pixels = np.column_stack([xs, ys]).astype(float)
    center = np.median(pixels, axis=0)
    # The largest connected component avoids distant same-color speckles.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    i = 1+int(np.argmax(stats[1:, 4]))
    ys, xs = np.where(labels == i)
    if len(xs) < 100:
        raise ValueError('Largest tape-colored region has too few pixels for wrist geometry')
    pixels = np.column_stack([xs, ys]).astype(float)
    center = np.mean(pixels, axis=0)
    values, vectors = np.linalg.eigh(np.cov(pixels.T))
    axis = vectors[:, -1]
    angle = float(np.arctan2(axis[1], axis[0]))
    clipped = bool(xs.min() <= 1 or xs.max() >= image.shape[1]-2 or ys.min() <= 1 or ys.max() >= image.shape[0]-2)
    ratio = float(values[-1]/max(values[0], 1.))
    z = None
    valid_fraction = 0.
    if depth is not None:
        samples = depth[ys, xs]
        valid = np.isfinite(samples) & (samples >= .035) & (samples <= .65)
        valid_fraction = float(valid.mean())
        if valid.sum() >= 30 and valid_fraction >= .5:
            z = float(np.median(samples[valid]))
    return {'center_px': center.tolist(), 'axis_rad': angle,
            'anisotropy': ratio, 'pixels': len(xs), 'clipped': clipped,
            'axis_reliable': ratio >= 2.5 and not clipped,
            'depth_m': z, 'valid_depth_fraction': valid_fraction,
            'bbox_xywh': [int(xs.min()), int(ys.min()), int(np.ptp(xs)+1), int(np.ptp(ys)+1)]}


def corrections(feature, reference, intrinsics, T_base_camera, max_translation=.003):
    """Small XY translation plus wrist-roll correction from current image error.

    The axis gain follows rotation of the camera about the gripper approach
    axis. The installed eye-in-hand rotation supplies the sign; recorded
    pregrasp->before-close pairs provide an independent empirical check.
    """
    if not feature.get('axis_reliable') or feature.get('depth_m') is None:
        raise ValueError('Wrist axis or depth is unreliable; keep the current hold')
    t = np.asarray(T_base_camera, float)
    error_px = np.asarray(reference['center_px'])-feature['center_px']
    z = feature['depth_m']
    u, v = feature['center_px']
    x, y = (u-intrinsics['ppx'])*z/intrinsics['fx'], (v-intrinsics['ppy'])*z/intrinsics['fy']
    projection = np.array([[intrinsics['fx']/z, 0, -intrinsics['fx']*x/z**2],
                           [0, intrinsics['fy']/z, -intrinsics['fy']*y/z**2]])
    jacobian = projection @ (-t[:3, :3].T[:, :2])
    # Damped image Jacobian: reduce unobservable corrections instead of
    # aborting the task at an arbitrary condition-number threshold.
    delta_xy = jacobian.T@np.linalg.solve(jacobian@jacobian.T+25.*np.eye(2), .35*error_px)
    length = np.linalg.norm(delta_xy)
    if length > max_translation:
        delta_xy *= max_translation/length
    axis_error = line_error(reference['axis_rad'], feature['axis_rad'])
    return np.r_[delta_xy, 0.], axis_error


def roll_step(axis_error, optical_axis_dot_tool_axis, limit=np.deg2rad(1.)):
    gain = -float(optical_axis_dot_tool_axis)
    if abs(gain) < .5:
        raise ValueError('Wrist optical axis cannot observe roll reliably')
    return float(np.clip(.35*axis_error/gain, -limit, limit))
