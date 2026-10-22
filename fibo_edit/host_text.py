# Text pass through the host's Qwen3-VL 4B (spec 4.5, 6). Duck-typed on the encoder so tests can use a fake one.
# No prompt weighting, no template strip, no padding: one prompt, one forward pass. Author: Stable Yogi.
import torch

from .taps import TAP_LAYERS, build_context

IMAGE_TOKEN = 151655
MAX_TOKENS = 4096  # ASSUMED: counted on the text tokens, before picture tokens are expanded
FULL_VISION = True  # ASSUMED: give the encoder Qwen3-VL's 3-axis positions and deepstack features (host helper)


def token_ids(tokenizer, text: str) -> list:
    return [int(t) for t in tokenizer([text])["input_ids"][0]][:MAX_TOKENS]


def segments(ids: list) -> list:
    """Split ids at every picture token: n pictures give n + 1 text runs."""
    runs, cur = [], []
    for t in ids:
        if t == IMAGE_TOKEN:
            runs.append(cur)
            cur = []
        else:
            cur.append(t)
    runs.append(cur)
    return runs


def splice(te, ids: list, pictures: list, device) -> tuple:
    """Embed text, put each picture's vision tokens where its picture token was. -> (embeds (1, L, D) f32, info)."""
    runs = segments(ids)
    if len(runs) - 1 != len(pictures):
        raise ValueError(f"{len(runs) - 1} picture tokens in the text but {len(pictures)} pictures")
    flat = [t for r in runs for t in r]
    table = te.get_input_embeddings()(torch.tensor([flat], device=device, dtype=torch.long)).float()
    parts, info, pos, at = [], [], 0, 0
    for i, run in enumerate(runs):
        if run:
            parts.append(table[:, at:at + len(run)])
            at += len(run)
            pos += len(run)
        if i < len(pictures):
            emb, extra = te.preprocess_embed({"type": "image", "data": pictures[i]}, device=device)
            emb = emb.reshape(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
            parts.append(emb)
            info.append({"type": "image", "index": pos, "size": emb.shape[1], "extra": extra})
            pos += emb.shape[1]
    return torch.cat(parts, dim=1), info


def _deepstack_hooks(te, mask, deepstack) -> list:
    """Add deepstack features to the picture positions after the first len(deepstack) blocks."""
    handles = []
    for i, feats in enumerate(deepstack or []):
        def hook(_mod, _args, out, feats=feats):
            x = out[0] if isinstance(out, tuple) else out
            x = x.clone()
            x[mask] = x[mask] + feats.to(device=x.device, dtype=x.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        handles.append(te.model.layers[i].register_forward_hook(hook))
    return handles


def run_encoder(te, embeds, info) -> torch.Tensor:
    """-> taps (15, L, D) in TAP_LAYERS order."""
    n = embeds.shape[1]
    kw = dict(embeds=embeds, attention_mask=torch.ones((1, n), device=embeds.device, dtype=torch.long), num_tokens=[n],
              embeds_info=info, intermediate_output=list(TAP_LAYERS), final_layer_norm_intermediate=False)
    handles = []
    if info and FULL_VISION and hasattr(te, "build_image_inputs"):
        position_ids, mask, deepstack = te.build_image_inputs(embeds, info)
        kw["position_ids"] = position_ids
        handles = _deepstack_hooks(te, mask, deepstack)
    try:
        out = te(None, **kw)
    finally:
        for h in handles:
            h.remove()
    z = out[1]
    if z is None or z.shape[1] != len(TAP_LAYERS):
        raise RuntimeError("the text encoder did not return the 15 hidden-state taps")
    return z[0]


def encode(te, tokenizer, text: str, pictures: list, device) -> torch.Tensor:
    """Full chat text + vision pictures (each (1, h, w, 3) 0..1) -> context (L, 46080) bf16."""
    ids = token_ids(tokenizer, text)
    embeds, info = splice(te, ids, pictures, device)
    return build_context(run_encoder(te, embeds, info))
