# The DiT as the host calls it: forward(x, timestep, context, ...) with references from the host channel.
# Imported only inside Forge. Author: Stable Yogi.
import torch

from backend import memory_management
from backend.args import dynamic_args

from . import dit
from .errors import OUT_OF_MEMORY, refuse

try:  # UNVERIFIED on GPU: the host's fast attention kernel, same (B, H, N, D) -> (B, N, H*D) contract
    from backend.attention import attention_function

    def _host_attention(q, k, v):
        return attention_function(q, k, v, q.shape[1], skip_reshape=True)

    dit.ATTEND["fn"] = _host_attention
except Exception as e:  # keep the plain kernel; say so once
    print(f"[FIBO Edit] host attention not found, using the plain kernel: {e}")


class FiboEditHostDiT(dit.FiboEditDiT):
    """Built by the host loader in place of its Krea2 class (scripts/fibo_register.py)."""

    def forward(self, x, timestep, context=None, transformer_options={}, **kwargs):
        dtype = getattr(self, "computation_dtype", None) or x.dtype
        refs = [r.to(device=x.device, dtype=dtype) for r in list(dynamic_args.ref_latents)]
        try:
            out = self.run(x, timestep, context, refs, dtype=dtype)
        except Exception as e:
            if memory_management.is_oom(e):
                memory_management.unload_all_models()
                memory_management.soft_empty_cache()
                refuse(OUT_OF_MEMORY.format(stage="sampling"))
            raise
        return out.to(x.dtype)
