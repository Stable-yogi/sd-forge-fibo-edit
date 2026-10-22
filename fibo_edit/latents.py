# 2x2 latent packing, token flattening and batch-norm style normalisation. Author: Stable Yogi.
import torch

NORM_EPS = 1e-4


def pack(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 4C, H/2, W/2); channel = c*4 + dy*2 + dx."""
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(b, c * 4, h // 2, w // 2)


def unpack(z: torch.Tensor) -> torch.Tensor:
    b, c4, h, w = z.shape
    z = z.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c4 // 4, h * 2, w * 2)


def to_tokens(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C), row-major."""
    return z.flatten(2).transpose(1, 2)


def from_tokens(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], h, w)


def _per_channel(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return v.to(device=like.device, dtype=torch.float32).reshape(1, -1, 1, 1)


def normalize(z: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    out = (z.float() - _per_channel(mean, z)) / torch.sqrt(_per_channel(var, z) + eps)
    return out.to(z.dtype)


def denormalize(z: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    out = z.float() * torch.sqrt(_per_channel(var, z) + eps) + _per_channel(mean, z)
    return out.to(z.dtype)
