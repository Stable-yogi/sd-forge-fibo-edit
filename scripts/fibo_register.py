# Registers FIBO 2.0 Edit with the Forge Neo loader at start-up. Never touches host files. Author: Stable Yogi.
# Each step runs on its own; a failed step prints one line and leaves Forge running without this model.
import os
import sys
import traceback

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

from fibo_edit import hooks  # noqa: E402
from fibo_edit.errors import TAG, say  # noqa: E402

REPO_DIR = os.path.join(EXT_ROOT, "fibo_edit", "repo")


def _log_fail(step: str):
    print(f"{TAG} {step} failed; FIBO 2.0 Edit is off, Forge is unaffected.\n" + traceback.format_exc())
    hooks.STATE["off"], hooks.STATE["why"] = True, step


def _register_arch():
    import torch

    import backend.loader as loader
    import huggingface_guess.detection as detection
    from huggingface_guess import model_list

    if getattr(loader, "_fibo_edit", False):
        return
    from fibo_edit.host_engine import FiboEditEngine

    class FiboEditBase(model_list.Flux2K4B):
        huggingface_repo = REPO_DIR
        unet_config = {"image_model": hooks.IMAGE_MODEL}
        sampling_settings = {"shift": 1.15}
        unet_extra_config = {}
        supported_inference_dtypes = [torch.bfloat16, torch.float32]
        memory_usage_factor = 20.0  # GUESSED: 8B DiT, long joint sequence; measure on GPU

        def clip_target(self, state_dict={}):
            return {"qwen3vl_4b.transformer": "text_encoder"}

    model_list.FiboEditBase = FiboEditBase
    model_list.models.insert(0, FiboEditBase)

    detection.detect_unet_config = hooks.make_detect(detection.detect_unet_config)

    import backend.nn.krea as krea_nn
    from fibo_edit.host_dit import FiboEditHostDiT

    hf = os.path.join(os.path.dirname(loader.__file__), "huggingface")
    loader.load_huggingface_component = hooks.make_lhc(
        loader.load_huggingface_component,
        is_ours=lambda guess: type(guess) is FiboEditBase,
        swap=lambda: hooks.swapped(krea_nn, "SingleStreamDiT", FiboEditHostDiT),
        te_repo=os.path.join(hf, "krea", "Krea-2-Raw"),
        vae_repo=os.path.join(hf, "black-forest-labs", "FLUX.2-klein-4B"),
    )

    FiboEditEngine.matched_guesses = [FiboEditBase]
    loader.possible_models = (FiboEditEngine,) + tuple(loader.possible_models)

    from backend.args import dynamic_args
    from backend.sampling import condition

    condition.ConditionCrossAttn.can_concat = hooks.make_can_concat(
        condition.ConditionCrossAttn.can_concat, lambda: bool(getattr(dynamic_args, "fibo_edit", False))
    )
    loader._fibo_edit = True
    say("FIBO 2.0 Edit registered.")


def _module_dirs(sub: str, opt: str):
    from modules import paths, shared

    dirs = [os.path.abspath(os.path.join(paths.models_path, sub))]
    dirs += [d for d in (getattr(shared.cmd_opts, opt, None) or []) if d]
    return dirs


def _register_loader():
    import backend.loader as loader
    import modules.sd_models as sdm
    from backend.args import dynamic_args

    if getattr(sdm.forge_loader, "_fibo_edit", False):
        return

    def on_load():
        dynamic_args.fibo_edit = False

    wrapped = hooks.make_loader(sdm.forge_loader, on_load, lambda: _module_dirs("text_encoder", "text_encoder_dirs"), lambda: _module_dirs("VAE", "vae_dirs"))
    sdm.forge_loader = wrapped
    loader.forge_loader = wrapped


for _step, _fn in (("registration", _register_arch), ("loader wrapper", _register_loader)):
    if hooks.STATE["off"]:
        break
    try:
        _fn()
    except Exception:
        _log_fail(_step)
