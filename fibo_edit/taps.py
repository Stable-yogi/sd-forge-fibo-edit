# Text-encoder hidden-state taps -> six 3-layer bundles -> context tensor (spec 6). Author: Stable Yogi.
import torch

TAP_LAYERS = (5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35)
BUNDLES = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_EPS = 1e-6


def tap_order() -> list:
    """Index into the sorted 15 taps for every bundle column group, in context order."""
    pos = {layer: i for i, layer in enumerate(TAP_LAYERS)}
    return [pos[layer] for bundle in BUNDLES for layer in bundle]


def tap_norm(x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    x = x.float()
    x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + TAP_EPS)
    return x.to(out_dtype)


def build_context(taps: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """taps (15, L, D) [or (B, 15, L, D)] in TAP_LAYERS order -> (L, 18*D) [or (B, L, 18*D)]."""
    batched = taps.dim() == 4
    if not batched:
        taps = taps[None]
    normed = tap_norm(taps, out_dtype)
    ctx = torch.cat([normed[:, i] for i in tap_order()], dim=-1)
    return ctx if batched else ctx[0]
