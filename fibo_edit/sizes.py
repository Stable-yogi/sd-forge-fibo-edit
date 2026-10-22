# Picture sizes and summary-token count. Pure. Author: Stable Yogi.
import math

TARGET_AREA = 1024 * 1024
VISION_MAX_SIDE = 768
VISION_MIN_PIXELS = 3136
VISION_MAX_PIXELS = 1605632


def trained_size(aspect: float, target: int = TARGET_AREA) -> tuple:
    """Spec 5.2: (W, H) multiples of 16, >= 64, area within 90..110% of target, least crop loss."""
    lo, hi = 0.9 * target, 1.1 * target
    best, best_key = None, None
    for w in range(64, int(hi // 64) + 16, 16):
        h_min = max(64, int(math.ceil(lo / w / 16)) * 16)
        for h in range(h_min, int(hi // w) + 1, 16):
            area = w * h
            if area < lo or area > hi:
                continue
            r = w / h
            loss = round(1 - min(r / aspect, aspect / r), 9)
            key = (loss, abs(area - target), -w)
            if best_key is None or key < best_key:
                best, best_key = (w, h), key
    return best


def trained_size_for(width: int, height: int) -> tuple:
    return trained_size(width / height)


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def cover_crop(width: int, height: int, tw: int, th: int) -> tuple:
    """Spec 5.4: resize size (cover, rounded up) and the crop box (left, top, right, bottom)."""
    scale = max(tw / width, th / height)
    nw, nh = max(tw, _ceil(width * scale)), max(th, _ceil(height * scale))
    left, top = (nw - tw) // 2, (nh - th) // 2
    return (nw, nh), (left, top, left + tw, top + th)


def vision_size(width: int, height: int, max_side: int = VISION_MAX_SIDE) -> tuple:
    """Spec 5.6 first half: shrink so the longest side is <= max_side, never enlarge."""
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    s = max_side / longest
    return max(1, round(width * s)), max(1, round(height * s))


def vision_grid(width: int, height: int, factor: int = 32, min_px: int = VISION_MIN_PIXELS, max_px: int = VISION_MAX_PIXELS) -> tuple:
    """Spec 5.6 second half: sides to multiples of 32, pixels kept in [min_px, max_px]."""
    w = max(factor, round(width / factor) * factor)
    h = max(factor, round(height / factor) * factor)
    if w * h > max_px:
        beta = math.sqrt(width * height / max_px)
        w = max(factor, math.floor(width / beta / factor) * factor)
        h = max(factor, math.floor(height / beta / factor) * factor)
    elif w * h < min_px:
        beta = math.sqrt(min_px / (width * height))
        w = math.ceil(width * beta / factor) * factor
        h = math.ceil(height * beta / factor) * factor
    return w, h


def vision_tokens(width: int, height: int) -> int:
    w, h = vision_grid(*vision_size(width, height))
    return (w * h) // 1024


def summary_count(text_len: int, low: int = 256, high: int = 512) -> int:
    """Spec 8.9."""
    ratio = min(max((text_len - 256) / 1792, 0.0), 1.0)
    g = math.ceil((256 + ratio * 256) / 64) * 64
    return int(min(max(g, low), high))
