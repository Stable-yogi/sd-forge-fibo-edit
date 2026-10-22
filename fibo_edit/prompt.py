# Prompt -> instruction + caption -> compact JSON -> chat turn. Pure, no host. Author: Stable Yogi.
import json
import math

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
TURN_HEAD = "<|im_start|>user\n"
TURN_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
SCORE_KEYS = ("aesthetic_score", "preference_score")
KEEP_PAIR = ("short_description", "text_render")
EMPTY_CAPTION = {
    "short_description": "",
    "objects": [],
    "background_setting": "",
    "lighting": {},
    "aesthetics": {},
    "photographic_characteristics": {},
    "style_medium": "",
    "context": "",
    "artistic_style": "",
}
_DROP = object()


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def is_blank(prompt) -> bool:
    return prompt is None or not str(prompt).strip()


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, (str, list, dict)) and len(v) == 0


def _prune(v):
    """Drop empty values at every depth; returns _DROP when v itself ends up empty."""
    if isinstance(v, dict):
        out = {}
        for k, item in v.items():
            item = _prune(item)
            if item is not _DROP:
                out[k] = item
        v = out
    elif isinstance(v, list):
        v = [x for x in (_prune(x) for x in v) if x is not _DROP]
    return _DROP if _is_empty(v) else v


def _empty_like(v):
    if isinstance(v, list):
        return []
    if isinstance(v, dict):
        return {}
    if isinstance(v, str):
        return ""
    return v


def _strip_scores(caption: dict) -> dict:
    out = {}
    for k, v in caption.items():
        if k in SCORE_KEYS:
            continue
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk not in SCORE_KEYS}
        out[k] = v
    return out


def clean_caption(caption: dict) -> dict:
    """Spec 4.4. Returns {} when nothing is left."""
    if len(caption) == 2 and set(caption) == set(KEEP_PAIR):
        out = {}
        for k in KEEP_PAIR:
            v = _prune(caption[k])
            out[k] = _empty_like(caption[k]) if v is _DROP else v
        return out
    pruned = _prune(_strip_scores(caption))
    if pruned is _DROP:
        return {}
    return {k: v for k, v in pruned.items() if v != {}}


def _as_text(v) -> str:
    return v if isinstance(v, str) else compact(v)


def parse_prompt(raw: str):
    """Returns (instruction: str, caption: None | str | dict)."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return raw, None
    if not isinstance(obj, dict) or "edit_instruction" not in obj:
        return raw, None
    instruction = _as_text(obj["edit_instruction"])
    if "structured_caption" in obj:
        caption = obj["structured_caption"]
    else:
        # F14 case 4: without structured_caption the other keys are the caption
        caption = {k: v for k, v in obj.items() if k != "edit_instruction"}
    if isinstance(caption, str):
        return instruction, caption
    if isinstance(caption, dict):
        caption = clean_caption(caption)
        return instruction, (caption or None)
    caption = _prune(caption)
    return instruction, (None if caption is _DROP else caption)


def image_keys(n: int) -> list:
    if n == 1:
        return ["image"]
    return [f"image_{i}" for i in range(1, n + 1)]


def wrap(instruction: str, caption, n_images: int) -> str:
    body = {k: VISION for k in image_keys(n_images)} if n_images > 0 else {}
    if caption is not None:
        body["structured_caption"] = caption
    body["edit_instruction"] = instruction
    return compact(body)


def chat_turn(text: str) -> str:
    return TURN_HEAD + text + TURN_TAIL


def build_text(raw: str, n_images: int) -> str:
    instruction, caption = parse_prompt(raw)
    return chat_turn(wrap(instruction, caption, n_images))


def empty_caption_turn() -> str:
    return chat_turn(compact(EMPTY_CAPTION))
