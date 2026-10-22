# Paint the masked area flat grey before any resize (spec 11.2). Author: Stable Yogi.
import numpy as np

from .errors import MASK_SIZE, FiboRefusal

GREY = 128
THRESHOLD = 128


def mask_to_grey_array(mask) -> np.ndarray:
    """PIL image or array -> (H, W) uint8."""
    if hasattr(mask, "convert"):
        mask = mask.convert("L")
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def paint_grey(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """image (H, W, 3) uint8, mask (H, W) uint8 -> copy with mask >= 128 painted (128,128,128)."""
    if image.shape[:2] != mask.shape[:2]:
        p = f"{image.shape[1]}x{image.shape[0]}"
        m = f"{mask.shape[1]}x{mask.shape[0]}"
        raise FiboRefusal(MASK_SIZE.format(m=m, p=p))
    out = np.array(image, dtype=np.uint8, copy=True)
    out[mask >= THRESHOLD] = GREY
    return out
