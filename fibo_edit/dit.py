# FIBO 2.0 Edit diffusion transformer, written from the behaviour spec. Author: Stable Yogi.
# Layers are created through torch.nn.<Name> looked up at build time, so the host's
# using_forge_operations() can swap in its own Linear (GGUF, fp8, offload) without changes here.
import math

import torch
import torch.nn.functional as F
from torch import nn

from . import rope
from .sizes import summary_count

HEAD_DIM = 128
NORM_EPS = 1e-5
FINAL_EPS = 1e-6
TIME_SCALE = 1000.0


def default_attention(q, k, v):
    """q, k, v (B, H, N, D) -> (B, N, H*D). The host replaces this with its fast kernel."""
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


ATTEND = {"fn": default_attention}


def cast(t: torch.Tensor, like: torch.Tensor, dtype=None) -> torch.Tensor:
    return t.to(device=like.device, dtype=dtype or like.dtype)


def time_sinus(v: torch.Tensor, width: int = 256) -> torch.Tensor:
    """v = 1000 * sigma -> [cos(v f_i)..., sin(v f_i)...], f_i = 10000^(-i/half)."""
    half = width // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=v.device) / half)
    args = v.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMS(nn.Module):
    def __init__(self, width: int, eps: float = NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * cast(self.weight, x32, torch.float32)).to(x.dtype)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, y):
        gate, up = self.w13(y).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class SplitFFN(nn.Module):
    """Same maths as FFN with separate gate (w1) and up (w3) matrices (spec 8.3, F4)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, y):
        return self.w2(F.silu(self.w1(y)) * self.w3(y))


def _heads(x, n):
    b, t, _ = x.shape
    return x.view(b, t, n, -1)


class SelfAttention(nn.Module):
    """Fused qkv, per-head RMS on q/k, rotary on q/k, full softmax."""

    def __init__(self, dim: int):
        super().__init__()
        self.n = dim // HEAD_DIM
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMS(HEAD_DIM)
        self.k_norm = RMS(HEAD_DIM)

    def forward(self, x, pe):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(_heads(q, self.n)).transpose(1, 2)
        k = self.k_norm(_heads(k, self.n)).transpose(1, 2)
        v = _heads(v, self.n).transpose(1, 2)
        q, k = rope.apply(q, pe), rope.apply(k, pe)
        return self.out(ATTEND["fn"](q, k, v))


class CrossAttention(nn.Module):
    """q from the stream, k / v from a context (to_kv rows: k then v). No rotary."""

    def __init__(self, dim: int, ctx_dim: int):
        super().__init__()
        self.n = dim // HEAD_DIM
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMS(HEAD_DIM)
        self.norm_k = RMS(HEAD_DIM)

    def forward(self, x, ctx):
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        q = self.norm_q(_heads(self.to_q(x), self.n)).transpose(1, 2)
        k = self.norm_k(_heads(k, self.n)).transpose(1, 2)
        v = _heads(v, self.n).transpose(1, 2)
        return self.to_out(ATTEND["fn"](q, k, v))


def _segments(s, spans, fn):
    """Apply fn(rows, mod) per span; spans = [(start, end, mod)]."""
    return torch.cat([fn(s[:, a:b], m) for a, b, m in spans], dim=1)


class Block(nn.Module):
    def __init__(self, dim: int, ffn: int, cond_dim: int, ctx_dim: int = 0):
        super().__init__()
        self.attention = SelfAttention(dim)
        self.attention_norm1, self.attention_norm2 = RMS(dim), RMS(dim)
        self.ffn_norm1, self.ffn_norm2 = RMS(dim), RMS(dim)
        self.feed_forward = FFN(dim, ffn)
        self.adaLN_modulation = nn.Sequential(nn.Linear(cond_dim, 4 * dim, bias=True))
        self.injects = ctx_dim > 0
        if self.injects:
            self.cross_attention = CrossAttention(dim, ctx_dim)
            self.cross_attn_norm1, self.cross_attn_norm2 = RMS(dim), RMS(dim)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def modulation(self, c):
        """c (B, cond) -> four (B, 1, dim) float32 slices."""
        return [m.float()[:, None] for m in self.adaLN_modulation(c).chunk(4, dim=-1)]

    def forward(self, s, spans, pe, ctx=None, dtype=None):
        dtype = dtype or s.dtype
        x = _segments(s, spans, lambda r, m: (self.attention_norm1(r).float() * (1 + m[0])).to(dtype))
        a = self.attention_norm2(self.attention(x, pe)).float()
        s = s + _segments(a, spans, lambda r, m: torch.tanh(m[1]) * r)
        if self.injects:
            z = self.cross_attention(self.cross_attn_norm1(s).to(dtype), ctx)
            gate = torch.tanh(cast(self.cross_attn_gate, s, torch.float32))
            s = s + gate * self.cross_attn_norm2(z).float()
        x = _segments(s, spans, lambda r, m: (self.ffn_norm1(r).float() * (1 + m[2])).to(dtype))
        f = self.ffn_norm2(self.feed_forward(x)).float()
        return s + _segments(f, spans, lambda r, m: torch.tanh(m[3]) * r)


class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, ffn: int, ctx_dim: int):
        super().__init__()
        self.cross_attn_norm = RMS(dim)
        self.cross_attention = CrossAttention(dim, ctx_dim)
        self.self_attn_norm = RMS(dim)
        self.self_attention = CrossAttention(dim, dim)
        self.ffn_norm = RMS(dim)
        self.ffn = FFN(dim, ffn)

    def forward(self, g, ctx, dtype):
        g = g + self.cross_attention(self.cross_attn_norm(g).to(dtype), ctx).float()
        h = self.self_attn_norm(g).to(dtype)
        g = g + self.self_attention(h, h).float()
        return g + self.ffn(self.ffn_norm(g).to(dtype)).float()


class Perceiver(nn.Module):
    def __init__(self, dim: int, ffn: int, ctx_dim: int, layers: int, queries: int):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(queries, dim))
        self.layers = nn.ModuleList([PerceiverLayer(dim, ffn, ctx_dim) for _ in range(layers)])
        self.norm_out = RMS(dim)

    def forward(self, ctx, count, dtype):
        g = cast(self.latent_queries[:count], ctx, torch.float32)[None].expand(ctx.shape[0], -1, -1)
        for layer in self.layers:
            g = layer(g, ctx, dtype)
        return self.norm_out(g)


class TimeEmbedder(nn.Module):
    def __init__(self, width: int, mid: int, out: int):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, mid, bias=True), nn.SiLU(), nn.Linear(mid, out, bias=True))

    def forward(self, sigma, dtype):
        return self.mlp(time_sinus(TIME_SCALE * sigma, self.width).to(dtype))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim, bias=True))
        self.linear = nn.Linear(dim, out_ch, bias=True)

    def forward(self, x, c, dtype):
        scale = self.adaLN_modulation(c).float()[:, None]
        x = F.layer_norm(x.float(), (x.shape[-1],), eps=FINAL_EPS) * (1 + scale)
        return self.linear(x.to(dtype))


class FiboEditDiT(nn.Module):
    def __init__(self, in_channels=128, dim=3584, n_layers=44, n_refiner_layers=2, injection_layers=(6, 14, 22, 30, 38),
                 text_bundle_width=7680, perceiver_layers=2, max_summary_tokens=512, ffn_dim=9728,
                 time_sinus_width=256, t_embed_mid=1024, t_embed_out=256, out_channels=None, **_ignored):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.injection_layers = tuple(injection_layers)
        self.n_bundles = 1 + len(self.injection_layers)
        self.bundle_width = text_bundle_width
        cond = t_embed_out
        self.x_embedder = nn.Linear(in_channels, dim, bias=True)
        self.t_embedder = TimeEmbedder(time_sinus_width, t_embed_mid, t_embed_out)
        self.noise_refiner = nn.ModuleList([Block(dim, ffn_dim, cond) for _ in range(n_refiner_layers)])
        self.layers = nn.ModuleList([Block(dim, ffn_dim, cond, text_bundle_width if i in self.injection_layers else 0) for i in range(n_layers)])
        self.perceiver = Perceiver(dim, ffn_dim, text_bundle_width, perceiver_layers, max_summary_tokens)
        self.final_layer = FinalLayer(dim, self.out_channels, cond)

    def _embed(self, lat, dtype):
        return self.x_embedder(lat.flatten(2).transpose(1, 2).to(dtype)).float()

    def run(self, x, sigma, context, refs=(), dtype=None):
        """x (B, C, H, W) noisy latent; sigma (B,) in 0..1; context (B|1, L, n_bundles*width); refs [(1, C, h, w)]."""
        dtype = dtype or x.dtype
        b, _, h, w = x.shape
        if context.shape[0] != b:
            context = context.expand(b, -1, -1)
        bundles = [c.to(dtype) for c in context.chunk(self.n_bundles, dim=-1)]
        sig = sigma.reshape(-1).to(device=x.device, dtype=torch.float32)
        if sig.shape[0] != b:
            sig = sig.expand(b)
        conds = self.t_embedder(torch.cat([sig, sig.new_zeros(1)]), dtype)
        c_n, c_0 = conds[:b], conds[b:]
        ref_hws = [tuple(r.shape[-2:]) for r in refs]
        img = self._embed(x, dtype)
        ref_tok = [self._embed(r, dtype).expand(b, -1, -1) for r in refs]
        hw = h * w
        g = summary_count(context.shape[1], high=self.perceiver.latent_queries.shape[0])

        # (b) refiner over [image, refs]
        pe = rope.table(rope.stream_ids(0, (h, w), ref_hws).to(x.device))
        s = torch.cat([img] + ref_tok, dim=1)
        for blk in self.noise_refiner:
            mod_n, mod_0 = blk.modulation(c_n), blk.modulation(c_0)
            s = blk(s, [(0, hw, mod_n), (hw, s.shape[1], mod_0)] if refs else [(0, hw, mod_n)], pe, dtype=dtype)

        # (c, d) summary tokens first, then the refined stream
        summ = self.perceiver(bundles[0], g, dtype).float()
        s = torch.cat([summ, s], dim=1)
        pe = rope.table(rope.stream_ids(g, (h, w), ref_hws).to(x.device))
        n_total = s.shape[1]
        k = 0
        for i, blk in enumerate(self.layers):
            mod_n, mod_0 = blk.modulation(c_n), blk.modulation(c_0)
            spans = [(0, g, mod_0), (g, g + hw, mod_n)] + ([(g + hw, n_total, mod_0)] if refs else [])
            ctx = None
            if blk.injects:
                k += 1
                ctx = bundles[k]
            s = blk(s, spans, pe, ctx, dtype=dtype)

        out = self.final_layer(s[:, g:g + hw], c_n, dtype)
        return out.transpose(1, 2).reshape(b, self.out_channels, h, w)


def split_ffns(model: nn.Module) -> nn.Module:
    """Replace every fused FFN with a SplitFFN holding the same numbers (F4)."""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, FFN):
                gate, up = child.w13.weight.data.chunk(2, dim=0)
                new = SplitFFN(gate.shape[1], child.hidden)
                new.w1.weight.data.copy_(gate)
                new.w3.weight.data.copy_(up)
                new.w2.weight.data.copy_(child.w2.weight.data)
                setattr(mod, child_name, new)
    return model
