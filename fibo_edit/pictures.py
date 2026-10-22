# Picture preparation and the refusal checks of one edit job (spec 5, 11, 12). Pure, PIL + numpy. Author: Stable Yogi.
import numpy as np
from PIL import Image

from . import errors
from .errors import FiboRefusal
from .mask import mask_to_grey_array, paint_grey
from .prompt import is_blank
from .sizes import cover_crop, trained_size_for, vision_size

MAX_PICTURES = 5
SIZE_STEP = 16
MIN_SIDE = 64
LANCZOS = Image.Resampling.LANCZOS


def check_count(n: int) -> None:
    if n < 1:
        raise FiboRefusal(errors.NO_PICTURE)
    if n > MAX_PICTURES:
        raise FiboRefusal(errors.TOO_MANY.format(n=n))


def check_prompt(text) -> None:
    if is_blank(text):
        raise FiboRefusal(errors.EMPTY_PROMPT)


def check_mask(n_pictures: int, mask_size, picture_size) -> None:
    """mask_size / picture_size are (w, h); mask_size None means no mask."""
    if mask_size is None:
        return
    if n_pictures != 1:
        raise FiboRefusal(errors.MASK_MANY)
    if tuple(mask_size) != tuple(picture_size):
        m, p = "%dx%d" % tuple(mask_size), "%dx%d" % tuple(picture_size)
        raise FiboRefusal(errors.MASK_SIZE.format(m=m, p=p))


def as_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def apply_mask(img: Image.Image, mask, invert: bool = False) -> Image.Image:
    """Spec 11.2 on the original picture: mask >= 128 painted flat grey. invert = host 'inpaint not masked'."""
    m = mask_to_grey_array(mask)
    if invert:
        m = 255 - m
    return Image.fromarray(paint_grey(np.asarray(as_rgb(img)), m))


def prepare(img: Image.Image) -> Image.Image:
    """Spec 5.4: RGB, Lanczos cover-resize to its own trained size, centre crop."""
    img = as_rgb(img)
    tw, th = trained_size_for(*img.size)
    (nw, nh), box = cover_crop(img.width, img.height, tw, th)
    if (nw, nh) != img.size:
        img = img.resize((nw, nh), LANCZOS)
    return img.crop(box) if box != (0, 0, nw, nh) else img


def vision_copy(img: Image.Image) -> Image.Image:
    """Spec 5.6 first half: longest side at most 768, aspect kept, never enlarged."""
    w, h = vision_size(*img.size)
    return img if (w, h) == img.size else img.resize((w, h), LANCZOS)


def to_unit_array(img: Image.Image) -> np.ndarray:
    """(1, H, W, 3) float32 in 0..1 (the host's BHWC picture layout)."""
    return (np.asarray(as_rgb(img), dtype=np.float32) / 255.0)[None]


def to_chw_array(img: Image.Image) -> np.ndarray:
    """(1, 3, H, W) float32 in 0..1, the layout the host VAE helper takes."""
    return np.ascontiguousarray(np.moveaxis(to_unit_array(img)[0], 2, 0)[None])


def snap_size(width: int, height: int) -> tuple:
    """Spec 15.3: a typed size floored to a multiple of 16, at least 64."""
    w = max(MIN_SIDE, int(width) // SIZE_STEP * SIZE_STEP)
    h = max(MIN_SIDE, int(height) // SIZE_STEP * SIZE_STEP)
    return w, h


def output_size(first: Image.Image, manual: bool, typed_w: int, typed_h: int) -> tuple:
    """Spec 5.3: trained size of picture 1 unless the manual size option is on."""
    if manual:
        return snap_size(typed_w, typed_h)
    return trained_size_for(*first.size)


def recommended(width: int, height: int) -> dict:
    """The panel button (spec 3.1): Euler, Simple, 4 steps, CFG 1 and picture 1's trained size."""
    out = {"sampler": "Euler", "scheduler": "Simple", "steps": 4, "cfg": 1.0, "width": None, "height": None}
    if width and height and width > 0 and height > 0:
        out["width"], out["height"] = trained_size_for(int(width), int(height))
    return out


def as_image(x):
    """The web page gives picture objects; the API gives base64 text. Both become a picture."""
    if isinstance(x, str) and x:
        import base64
        import io

        data = x.split(",", 1)[1] if x.startswith("data:") else x
        return Image.open(io.BytesIO(base64.b64decode(data)))
    return x


def collect(first, extras) -> list:
    """Picture 1 then the filled extra slots, in slot order."""
    pics = [first] if first is not None else []
    return pics + [as_image(x) for x in (extras or []) if x is not None and x != ""]
