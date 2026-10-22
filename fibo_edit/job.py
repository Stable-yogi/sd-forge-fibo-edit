# One edit job, prepared from what the panel and the img2img tab hold. Pure (PIL + numpy). Author: Stable Yogi.
import numpy as np
from PIL import Image

from . import errors, pictures

READY = "Ready: FIBO 2.0 Edit is loaded."


def raw_prompt(prompt) -> str:
    """The prompt as typed (the host may hand a list for API batches)."""
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ""
    return "" if prompt is None else str(prompt)


def mask_from_host(mask):
    """The host's inpaint mask as 8-bit grey: a painted RGBA layer uses its alpha, anything else its grey value."""
    if mask is None:
        return None
    if getattr(mask, "mode", None) == "RGBA" and mask.getextrema()[-1] != (255, 255):
        return mask.split()[-1]
    return mask.convert("L") if hasattr(mask, "convert") else Image.fromarray(np.asarray(mask, dtype=np.uint8))


def build_job(first, extras, prompt, mask=None, invert=False, manual=False, typed_w=0, typed_h=0) -> dict:
    """-> {"pictures": cropped RGB pictures, "vision": (1,h,w,3) arrays, "width", "height", "prompt"}. Refuses on bad input."""
    pics = [pictures.as_rgb(p) for p in pictures.collect(first, extras)]
    pictures.check_count(len(pics))
    text = raw_prompt(prompt)
    pictures.check_prompt(text)
    grey = mask_from_host(mask)
    if grey is not None:
        pictures.check_mask(len(pics), grey.size, pics[0].size)
        pics[0] = pictures.apply_mask(pics[0], grey, invert=invert)
    prepared = [pictures.prepare(p) for p in pics]
    w, h = pictures.output_size(prepared[0], manual, typed_w, typed_h)
    vision = [pictures.to_unit_array(pictures.vision_copy(p)) for p in prepared]
    return {"pictures": prepared, "vision": vision, "width": int(w), "height": int(h), "prompt": text}


def status_text(is_fibo: bool, off: bool, why: str = "", last: str = "") -> str:
    if off:
        return errors.DISABLED.format(why=why or "see the console")
    if not is_fibo:
        return errors.INERT
    return last or READY
