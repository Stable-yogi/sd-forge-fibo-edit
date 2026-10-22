# Weight-file sniffing and the text encoder / VAE pick for a bare FIBO file. No host, no tensor data. Author: Stable Yogi.
import json
import os
import struct

from . import gguf_names, weights

EXTS = (".safetensors", ".sft", ".gguf")
TE_WORDS = ("qwen3vl_4b", "qwen3-vl-4b", "qwen3_vl_4b")
TE_MARK = "model.visual.deepstack_merger_list.0.norm.weight"
VAE_MARKS = ("decoder.post_quant_conv.weight", "bn.running_mean")


def safetensors_keys(path: str) -> list:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 1 << 28:
            raise ValueError("bad safetensors header")
        head = json.loads(f.read(n))
    return [k for k in head if k != "__metadata__"]


def tensor_names(path: str) -> list:
    low = str(path).lower()
    if low.endswith(".gguf"):
        return list(gguf_names.read_header(path)["tensors"])
    if low.endswith((".safetensors", ".sft")):
        return safetensors_keys(path)
    return []


def is_bare_fibo(path) -> bool:
    """True when the file holds the FIBO DiT and no text encoder of its own. Never raises."""
    try:
        names = tensor_names(str(path))
    except Exception:
        return False
    return weights.is_fibo(names) and not any(n.startswith("text_encoders.") for n in names)


def _files(dirs) -> list:
    """Weight files under each folder, sub-folders included (the host lists its module folders the same way)."""
    out, seen = [], set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, subdirs, names in os.walk(d):
            subdirs.sort()
            for n in sorted(names):
                p = os.path.join(root, n)
                if n.lower().endswith(EXTS) and p.lower() not in seen:
                    seen.add(p.lower())
                    out.append(p)
    return out


def _has_key(path: str, marks) -> bool:
    try:
        keys = set(safetensors_keys(path)) if path.lower().endswith((".safetensors", ".sft")) else set()
    except Exception:
        return False
    return any(m in keys for m in marks)


def is_te_name(name: str) -> bool:
    low = os.path.basename(str(name)).lower()
    return any(w in low for w in TE_WORDS)


def is_vae_name(name: str) -> bool:
    low = os.path.basename(str(name)).lower()
    return "flux2" in low and "vae" in low


def find_text_encoder(dirs):
    named = [p for p in _files(dirs) if is_te_name(p)]
    if named:
        return named[0]
    return next((p for p in _files(dirs) if _has_key(p, (TE_MARK,))), None)


def find_vae(dirs):
    named = [p for p in _files(dirs) if is_vae_name(p)]
    if named:
        return named[0]
    return next((p for p in _files(dirs) if _has_key(p, VAE_MARKS)), None)


def foreign_vae(given):
    """The first given file that is a VAE of another family (it has decoder and encoder tensors but no FLUX.2 mark), else None."""
    for g in given or []:
        g = str(g)
        if is_vae_name(g) or is_te_name(g):
            continue
        try:
            keys = tensor_names(g)
        except Exception:
            continue
        if any(m in keys for m in VAE_MARKS):
            continue
        if any(k.startswith("decoder.") for k in keys) and any(k.startswith("encoder.") for k in keys):
            return g
    return None


def plan_modules(given, te_dirs, vae_dirs) -> dict:
    """given: the user's extra module paths. Returns {"add": [...], "missing": [(what, dirs)]}."""
    given = [str(g) for g in (given or [])]
    add, missing = [], []
    if not any(is_te_name(g) or _has_key(g, (TE_MARK,)) for g in given):
        te = find_text_encoder(te_dirs)
        if te:
            add.append(te)
        else:
            missing.append(("Qwen3-VL 4B text encoder", list(te_dirs)))
    if not any(is_vae_name(g) or _has_key(g, VAE_MARKS) for g in given):
        vae = find_vae(vae_dirs)
        if vae:
            add.append(vae)
        else:
            missing.append(("FLUX.2 VAE", list(vae_dirs)))
    return {"add": add, "missing": missing}
