# Rotary position ids, angle tables and their use on q / k. Author: Stable Yogi.
import torch

AXES = (32, 48, 48)
THETA = 256.0
IMAGE_PLANE = 513


def summary_ids(count: int) -> torch.Tensor:
    ids = torch.zeros(count, 3)
    ids[:, 0] = torch.arange(1, count + 1, dtype=torch.float32)
    return ids


def grid_ids(plane: int, h: int, w: int) -> torch.Tensor:
    ids = torch.zeros(h, w, 3)
    ids[..., 0] = plane
    ids[..., 1] = torch.arange(h, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w, dtype=torch.float32)[None, :]
    return ids.reshape(h * w, 3)


def stream_ids(summary: int, image_hw: tuple, ref_hws: list) -> torch.Tensor:
    """Spec 8.10: summary (j,0,0) j>=1, image (513,y,x), reference k (514+k,y,x)."""
    parts = [summary_ids(summary)] if summary else []
    parts.append(grid_ids(IMAGE_PLANE, *image_hw))
    parts += [grid_ids(IMAGE_PLANE + 1 + k, h, w) for k, (h, w) in enumerate(ref_hws)]
    return torch.cat(parts)


def _axis_table(pos: torch.Tensor, width: int, theta: float) -> torch.Tensor:
    exps = torch.arange(0, width, 2, dtype=torch.float64, device=pos.device) / width
    freqs = 1.0 / (theta ** exps)
    ang = (pos.to(torch.float64)[..., None] * freqs).to(torch.float32)
    cos, sin = torch.cos(ang), torch.sin(ang)
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*ang.shape, 2, 2)


def table(ids: torch.Tensor, axes=AXES, theta: float = THETA) -> torch.Tensor:
    """ids (B, N, 3) or (N, 3) -> (B, 1, N, sum(axes)/2, 2, 2) laid out [[cos,-sin],[sin,cos]]."""
    if ids.dim() == 2:
        ids = ids[None]
    parts = [_axis_table(ids[..., i], d, theta) for i, d in enumerate(axes)]
    return torch.cat(parts, dim=-3).unsqueeze(1)


def apply(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """x (B, H, N, D); pairs (2i, 2i+1) rotated. Math in float32, result in x.dtype."""
    xp = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    pe = pe.to(device=x.device, dtype=torch.float32)
    out = pe[..., 0] * xp[..., 0] + pe[..., 1] * xp[..., 1]
    return out.reshape(x.shape).to(x.dtype)
