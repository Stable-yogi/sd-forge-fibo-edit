# Wrappers the register script puts around host functions. Duck-typed, no host import, so tests use fakes.
# Rule: an unexpected error in our code turns the extension off and hands the call back to the host.
# Author: Stable Yogi.
import contextlib
import os

from . import files, weights
from .errors import DISABLED, FILE_MISSING, WRONG_VAE, FiboRefusal, refuse, say

IMAGE_MODEL = "fibo_edit"
TRANSFORMER_CLS = "FiboEditTransformer2DModel"
HOST_TRANSFORMER_CLS = "Krea2Transformer2DModel"  # the host branch whose GGUF/quant build path we reuse

STATE = {"off": False, "why": ""}


def disable(why: str) -> None:
    STATE["off"], STATE["why"] = True, str(why)
    say(DISABLED.format(why=why))


def is_off() -> bool:
    return STATE["off"]


def guarded(stage: str, fn, *args, **kwargs):
    """Run one of our steps. Refusal: printed, re-raised (stops the job). Anything else: extension off, None."""
    if STATE["off"]:
        return None
    try:
        return fn(*args, **kwargs)
    except FiboRefusal as e:
        say(str(e))
        raise
    except Exception as e:  # never crash the host
        disable(f"{stage}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------- detection
def detect_config(state_dict, key_prefix: str):
    """The host's unet config for a FIBO file, or None. Shapes come from .shape (GGUF tensors report the real shape)."""
    n = len(key_prefix)
    names = [k[n:] for k in state_dict if k.startswith(key_prefix)]
    inner = weights.find_prefix(names)
    if inner is None:
        return None
    m = len(inner)
    shapes = {k[m:]: tuple(state_dict[key_prefix + k].shape) for k in names if k.startswith(inner) and not weights.is_ignored(k)}
    return {"image_model": IMAGE_MODEL, **weights.config_from_shapes(shapes)}


def make_detect(orig):
    def _detect(state_dict, key_prefix):
        if not STATE["off"]:
            try:
                cfg = detect_config(state_dict, key_prefix)
                if cfg is not None:
                    return cfg
            except Exception as e:
                disable(f"detection: {type(e).__name__}: {e}")
        return orig(state_dict, key_prefix)

    _detect._fibo_edit = True
    return _detect


# ---------------------------------------------------------------- component build
@contextlib.contextmanager
def swapped(module, name: str, value):
    """Set module.name = value for the duration of the block, always restored."""
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


def drop_ignored(state_dict) -> int:
    if not isinstance(state_dict, dict):
        return 0
    gone = [k for k in state_dict if weights.is_ignored(k)]
    for k in gone:
        state_dict.pop(k)
    return len(gone)


def make_lhc(orig, is_ours, swap, te_repo: str, vae_repo: str):
    """swap(): context manager that puts our DiT class where the host's Krea2 branch builds its class.
    te_repo / vae_repo: host config folders whose text_encoder, tokenizer and vae sub-folders match our parts."""

    def _lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not is_ours(guess):
            return orig(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        if cls_name == TRANSFORMER_CLS:
            drop_ignored(state_dict)
            with swap():
                return orig(guess, component_name, lib_name, HOST_TRANSFORMER_CLS, repo_path, state_dict)
        if component_name in ("text_encoder", "tokenizer"):
            repo_path = te_repo
        elif component_name == "vae":
            repo_path = vae_repo
        return orig(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    _lhc._fibo_edit = True
    return _lhc


# ---------------------------------------------------------------- conditioning batch
def make_can_concat(orig, active):
    """active(): True while a FIBO model is loaded. Different text lengths never share a batch then."""

    def _can_concat(self, other):
        if active() and tuple(self.cond.shape) != tuple(other.cond.shape):
            return False
        return orig(self, other)

    _can_concat._fibo_edit = True
    return _can_concat


# ---------------------------------------------------------------- whole-model load
def extra_modules(sd_path, given, te_dirs, vae_dirs) -> list:
    """For a bare FIBO file: the text encoder and VAE files to add. Refuses when one is not found."""
    if not files.is_bare_fibo(sd_path):
        return []
    bad = files.foreign_vae(given)
    if bad:
        refuse(WRONG_VAE.format(name=os.path.basename(bad)))
    plan = files.plan_modules(given, te_dirs, vae_dirs)
    if plan["missing"]:
        what, where = plan["missing"][0]
        refuse(FILE_MISSING.format(what=what, where="; ".join(where) or "(no folder)"))
    if plan["add"]:
        say("bare model file; adding " + ", ".join(os.path.basename(p) for p in plan["add"]))
    return plan["add"]


def make_loader(orig, on_load, te_dirs, vae_dirs):
    """on_load(): called first on every model load (clears our flag). te_dirs()/vae_dirs(): folder lists."""

    def _forge_loader(sd, additional_state_dicts=None):
        on_load()
        given = list(additional_state_dicts or [])
        if not STATE["off"]:
            try:
                add = extra_modules(sd, given, te_dirs(), vae_dirs())
                if add:
                    additional_state_dicts = add + given
            except FiboRefusal:
                raise
            except Exception as e:
                disable(f"module pick: {type(e).__name__}: {e}")
        return orig(sd, additional_state_dicts=additional_state_dicts)

    _forge_loader._fibo_edit = True
    return _forge_loader
