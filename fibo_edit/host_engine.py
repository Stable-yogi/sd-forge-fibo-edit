# FIBO 2.0 Edit diffusion engine for Forge Neo. Imported only inside Forge. Author: Stable Yogi.
# The panel (scripts/fibo_edit_panel.py) starts a job: pictures, vision copies, the raw prompt.
# References are VAE-encoded through encode_first_stage with dynamic_args.is_referencing set
# (the channel the host's other edit engines use) and reach the DiT as dynamic_args.ref_latents.
import numpy as np
import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import errors, hooks, host_text, prompt
from .errors import refuse
from .schedule import MODE_SIZE, mu_for

OTHER_MODEL_FLAGS = ("kontext", "edit", "klein", "krea2", "wan", "pid", "anima", "moss")


class _MeanPosterior:
    """Spec 7.3: encode with the posterior mean. Swaps the host's sample() for mode() for one call."""

    def __enter__(self):
        import backend.nn.vae as vae_nn

        self.cls = vae_nn.DiagonalGaussianDistribution
        self.saved = self.cls.sample
        self.cls.sample = self.cls.mode
        return self

    def __exit__(self, *exc):
        self.cls.sample = self.saved
        return False


class FiboEditEngine(ForgeDiffusionEngine):
    matched_guesses = []  # filled by scripts/fibo_register.py

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})
        vae = VAE(model=huggingface_components["vae"], is_flux2=True)
        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=self._get_predictor(), config=estimated_config)
        self.text_encoder = clip.cond_stage_model.qwen3vl_4b
        self.tokenizer = clip.tokenizer.qwen3vl_4b
        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()
        self.is_fibo_edit = True
        self.shift_mode = MODE_SIZE
        for flag in OTHER_MODEL_FLAGS:  # spec 13.3: file-name words must not switch on another model's path
            setattr(dynamic_args, flag, False)
        dynamic_args.fibo_edit = True  # read by the can_concat patch (spec 13.4)
        self.fibo_reset()

    # ---------------------------------------------------------------- job state
    def fibo_reset(self):
        self.job_active = False
        self.job_prompt = None
        self.job_vision = []
        self.fibo_refs = []  # own list: the host's clear_references() empties self.ref_latents
        dynamic_args.ref_latents = []

    def fibo_start(self, raw_prompt: str, vision_pictures: list):
        """vision_pictures: (1, h, w, 3) float 0..1 arrays or tensors, in picture order."""
        self.fibo_reset()
        self.job_active = True
        self.job_prompt = raw_prompt
        self.job_vision = [torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for v in vision_pictures]

    def publish_refs(self):
        """Put this job's reference latents on the host channel the DiT reads (republished before each use)."""
        dynamic_args.ref_latents = list(self.fibo_refs)

    # ---------------------------------------------------------------- shift
    def set_shift(self, shift: float, *args, **kwargs):
        """The host passes its shift slider here (newer hosts add width and height); this model sets its own shift from the size instead."""
        return

    def apply_size_shift(self, width: int, height: int, mode: str = None):
        mu = mu_for(int(width), int(height), mode or self.shift_mode)
        self.forge_objects.unet.model.predictor.set_parameters(shift=mu)
        return mu

    # ---------------------------------------------------------------- VAE
    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        """x (B, 3, H, W) in -1..1 -> normalised packed latent (B, 128, H/16, W/16)."""
        pixels = x.movedim(1, -1).mul(0.5).add(0.5)
        with _MeanPosterior():
            sample = self.forge_objects.vae.encode(pixels)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        if dynamic_args.is_referencing:
            for i in range(sample.shape[0]):
                self.fibo_refs.append(sample[i:i + 1].detach().clone().cpu())
            self.publish_refs()
        return sample.to(x)

    # ---------------------------------------------------------------- text
    @torch.inference_mode()
    def get_learned_conditioning(self, texts):
        negative = bool(getattr(texts, "is_negative_prompt", False))
        if hooks.is_off():
            refuse(errors.DISABLED.format(why=hooks.STATE["why"]))
        if not negative:
            if not self.job_active:
                refuse(errors.PANEL_OFF)
            if not self.job_vision:
                refuse(errors.NO_PICTURE)
            if prompt.is_blank(self.job_prompt):
                refuse(errors.EMPTY_PROMPT)
            self.publish_refs()
            text = prompt.build_text(self.job_prompt, len(self.job_vision))
            pictures = self.job_vision
        else:  # spec 9.7: the empty-caption turn, no pictures in the text; the DiT still sees the references
            text, pictures = prompt.empty_caption_turn(), []
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        device = memory_management.text_encoder_device()
        try:
            ctx = host_text.encode(self.text_encoder, self.tokenizer, text, [p.to(device) for p in pictures], device)
        except Exception as e:
            if memory_management.is_oom(e):
                memory_management.unload_all_models()
                memory_management.soft_empty_cache()
                refuse(errors.OUT_OF_MEMORY.format(stage="text encoding"))
            raise
        return [ctx.cpu() for _ in texts]

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, text):
        try:
            n = len(host_text.token_ids(self.tokenizer, prompt.build_text(text or "", max(1, len(self.job_vision)))))
        except Exception:
            n = 0
        return n, max(host_text.MAX_TOKENS, n)
