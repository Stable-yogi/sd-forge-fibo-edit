# Sigma schedule and Euler step (rectified flow). Pure. Author: Stable Yogi.
import math

FIXED_MU = 1.15
MODE_SIZE = "size-based"
MODE_FIXED = "fixed 1.15"
MODES = (MODE_SIZE, MODE_FIXED)


def latent_tokens(width: int, height: int) -> int:
    return (width // 16) * (height // 16)


def mu_for_tokens(n: int) -> float:
    """Spec 9.3, not clamped."""
    return 0.5 + (1.15 - 0.5) / (4096 - 256) * (n - 256)


def mu_for(width: int, height: int, mode: str = MODE_SIZE) -> float:
    if mode == MODE_FIXED:
        return FIXED_MU
    return mu_for_tokens(latent_tokens(width, height))


def shift(sigma: float, mu: float) -> float:
    if sigma <= 0:
        return 0.0
    e = math.exp(mu)
    return e / (e + 1.0 / sigma - 1.0)


def rational_shift(sigma: float, s: float) -> float:
    """Reference only (F7 sigmas_rational_*); not used for sampling."""
    return s * sigma / (1 + (s - 1) * sigma)


def base_sigmas(steps: int) -> list:
    return [1.0 - i / steps for i in range(steps)] + [0.0]


def sigmas(width: int, height: int, steps: int = 4, mode: str = MODE_SIZE) -> list:
    mu = mu_for(width, height, mode)
    return [shift(s, mu) for s in base_sigmas(steps)]


def euler_step(x, v, sigma: float, sigma_next: float):
    return x + (sigma_next - sigma) * v
