"""CPU masks plus metric depth. Outputs a visible-surface point, not object COM."""
import cv2
import numpy as np


def hsv_bounds(kind, low=None, high=None):
    """OpenCV hue is 0..179. Explicit bounds override detector defaults."""
    defaults = {'orange_ball': ((3, 100, 70), (25, 255, 255)),
                'blue_tape': ((90, 60, 45), (135, 255, 255))}
    default_low, default_high = defaults.get(kind, ((0, 120, 80), (10, 255, 255)))
    return (low if low is not None else default_low, high if high is not None else default_high)


def detect_mask(bgr, kind='aruco', marker_id=0, hsv_low=None,
                hsv_high=None, roi=None, min_area_px=80, all_candidates=False):
    if all_candidates and kind != 'blue_tape':
        raise ValueError('Candidate masks are supported only for blue tape')
    height, width = bgr.shape[:2]
    region = np.zeros((height, width), np.uint8)
    if roi is None:
        region[:] = 255
    else:
        x, y, w, h = map(int, roi)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError('ROI is outside image')
        region[y:y+h, x:x+w] = 255
    if kind == 'aruco':
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(bgr)
        matches = [] if ids is None else [c for c, i in zip(corners, ids.ravel()) if i == marker_id]
        if len(matches) != 1:
            return None
        mask = np.zeros_like(region)
        cv2.fillConvexPoly(mask, np.round(matches[0][0]).astype(np.int32), 255)
        # Reject partially out-of-ROI targets rather than moving their centroid.
        if np.any((mask > 0) & (region == 0)):
            return None
    elif kind in ('hsv', 'orange_ball', 'blue_tape'):
        low, high = map(np.array, hsv_bounds(kind, hsv_low, hsv_high))
        if (low.shape != (3,) or high.shape != (3,) or np.any(low > high)
                or np.any(low < 0) or np.any(high > [179, 255, 255])):
            raise ValueError('HSV bounds must satisfy 0 <= low <= high <= (179,255,255)')
        mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), low.astype('uint8'), high.astype('uint8'))
        mask = cv2.bitwise_and(mask, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted([c for c in contours if cv2.contourArea(c) >= min_area_px],
                          key=cv2.contourArea, reverse=True)
        if kind == 'blue_tape':
            # A small upright roll can be similar in area to a cable at the image
            # edge. Exclude clipped candidates before checking target ambiguity.
            rx, ry, rw, rh = (0, 0, width, height) if roi is None else map(int, roi)
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x > rx and y > ry and x+w < rx+rw and y+h < ry+rh:
                    candidates.append(contour)
            contours = candidates
            if all_candidates:
                # Multi-view association, rather than largest-area selection,
                # decides which of these blue regions belongs to the object.
                masks = []
                if len(contours) > 64: return []  # Crowded scene: no arbitrary truncation.
                for contour in contours:
                    candidate = np.zeros_like(mask)
                    cv2.drawContours(candidate, [contour], -1, 255, -1)
                    candidate = cv2.erode(cv2.bitwise_and(candidate, mask), np.ones((3, 3), np.uint8))
                    if np.count_nonzero(candidate) >= min_area_px: masks.append(candidate)
                return masks
        if kind == 'orange_ball':
            candidates = []
            rx, ry, rw, rh = (0, 0, width, height) if roi is None else map(int, roi)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # A clipped silhouette biases position and may look circular.
                if x <= rx or y <= ry or x + w >= rx + rw or y + h >= ry + rh:
                    continue
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                _, radius = cv2.minEnclosingCircle(contour)
                circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
                occupancy = area / max(np.pi * radius ** 2, 1)
                if .75 <= w / h <= 1.33 and circularity >= .78 and occupancy >= .80:
                    candidates.append(contour)
            contours = candidates
        if not contours or (len(contours) > 1 and cv2.contourArea(contours[1]) > 0.4 * cv2.contourArea(contours[0])):
            return None  # ambiguous target: require a tighter ROI or unique marker
        color_mask = mask.copy()
        mask[:] = 0
        cv2.drawContours(mask, [contours[0]], -1, 255, -1)
        if kind == 'blue_tape':
            # Tape may be a roll: never fill its hole with background depth.
            mask = cv2.bitwise_and(mask, color_mask)
    else:
        raise ValueError('Detector must be aruco, hsv, orange_ball, or blue_tape')
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return mask if np.count_nonzero(mask) >= min_area_px else None


def depth_measurement(mask, depth_m, deproject, min_depth_m=0.1, max_depth_m=1.5,
                      min_valid_fraction=0.5, max_depth_mad_m=0.02):
    if mask is None:
        return None, {'reason': 'target_not_unique_or_not_visible'}
    if mask.shape != depth_m.shape:
        raise ValueError('Depth must be aligned to the mask image')
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, {'reason': 'empty_mask'}
    values = depth_m[ys, xs]
    valid = np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
    fraction = float(valid.mean())
    if valid.sum() < 20 or fraction < min_valid_fraction:
        return None, {'reason': 'insufficient_depth', 'valid_fraction': fraction}
    xs, ys, values = xs[valid], ys[valid], values[valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > max_depth_mad_m:
        return None, {'reason': 'depth_spread', 'depth_mad_m': mad}
    # Reject depth-edge/background pixels before choosing the representative ray.
    good = np.abs(values - median) <= max(3 * mad, 0.005)
    pixel = [float(np.median(xs[good])), float(np.median(ys[good]))]
    point = np.asarray(deproject(pixel, median), float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None, {'reason': 'invalid_deprojection'}
    return point, {'reason': None, 'pixel': pixel, 'depth_m': median,
                   'depth_mad_m': mad, 'valid_fraction': fraction,
                   'position_semantics': 'visible_surface_proxy'}
