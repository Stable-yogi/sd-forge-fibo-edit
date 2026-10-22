# Weight names: fingerprint, filtering, model config from tensor shapes, npz loader. Author: Stable Yogi.
import re

FINGERPRINT = ("perceiver.latent_queries", "noise_refiner.0.adaLN_modulation.0.weight", "layers.6.cross_attn_gate")
PREFIXES = ("", "model.diffusion_model.", "diffusion_model.", "transformer.", "model.")
IGNORED_SUFFIXES = ("comfy_quant",)  # ASSUMED marker name for comfy quant config tensors


def is_ignored(name: str) -> bool:
    return name.endswith(IGNORED_SUFFIXES)


def find_prefix(names) -> str | None:
    """Return the key prefix under which all fingerprint tensors exist, or None."""
    names = set(names)
    for p in PREFIXES:
        if all(p + k in names for k in FINGERPRINT):
            return p
    return None


def is_fibo(names) -> bool:
    return find_prefix(names) is not None


def strip(state: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in state.items() if k.startswith(prefix) and not is_ignored(k)}


def _count(names, pattern: str) -> list:
    rx = re.compile(pattern)
    return sorted({int(m.group(1)) for k in names for m in [rx.fullmatch(k)] if m})


def config_from_shapes(shapes: dict) -> dict:
    """shapes: name -> torch shape (out, in). Returns kwargs for FiboEditDiT."""
    s = shapes
    return {
        "in_channels": int(s["x_embedder.weight"][1]),
        "dim": int(s["x_embedder.weight"][0]),
        "n_layers": len(_count(s, r"layers\.(\d+)\.attention\.qkv\.weight")),
        "n_refiner_layers": len(_count(s, r"noise_refiner\.(\d+)\.attention\.qkv\.weight")),
        "injection_layers": _count(s, r"layers\.(\d+)\.cross_attn_gate"),
        "text_bundle_width": int(s["perceiver.layers.0.cross_attention.to_kv.weight"][1]),
        "perceiver_layers": len(_count(s, r"perceiver\.layers\.(\d+)\.ffn_norm\.weight")),
        "max_summary_tokens": int(s["perceiver.latent_queries"][0]),
        "ffn_dim": int(s["layers.0.feed_forward.w2.weight"][1]),
        "time_sinus_width": int(s["t_embedder.mlp.0.weight"][1]),
        "t_embed_mid": int(s["t_embedder.mlp.0.weight"][0]),
        "t_embed_out": int(s["t_embedder.mlp.2.weight"][0]),
        "out_channels": int(s["final_layer.linear.weight"][0]),
    }


def shapes_of(state: dict) -> dict:
    return {k: tuple(v.shape) for k, v in state.items()}


def load_npz(path: str) -> dict:
    import numpy as np
    import torch

    with np.load(path) as data:
        return {k: torch.from_numpy(np.array(data[k])) for k in data.files}


def build_from_state(state: dict, split_ffn: bool = False):
    """Plain-torch build (tests, CPU): config from shapes, then load the numbers."""
    from .dit import FiboEditDiT, split_ffns

    model = FiboEditDiT(**config_from_shapes(shapes_of(state)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise KeyError(f"missing {missing[:5]} unexpected {unexpected[:5]}")
    return split_ffns(model) if split_ffn else model
