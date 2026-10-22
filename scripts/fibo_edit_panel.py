# The "FIBO 2.0 Edit" panel in the img2img tab. Author: Stable Yogi.
# Every hook goes through hooks.guarded: a refusal stops the job with one line, any other error turns
# the extension off and lets Forge carry on.
import os
import sys

import gradio as gr
import torch

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

from modules import scripts, shared  # noqa: E402
from modules.ui_components import InputAccordion  # noqa: E402

from fibo_edit import errors, hooks, job, pictures  # noqa: E402
from fibo_edit.schedule import MODE_SIZE, MODES  # noqa: E402

EXTRA_SLOTS = 4
LICENCE = "Model licence: non-commercial (CC BY-NC 4.0)."
TARGETS = ("img2img_sampling", "img2img_scheduler", "img2img_steps", "img2img_cfg_scale", "img2img_width", "img2img_height")


def _engine(model):
    return model if getattr(model, "is_fibo_edit", False) else None


def _status():
    return job.status_text(_engine(getattr(shared, "sd_model", None)) is not None, hooks.is_off(), hooks.STATE["why"], errors.LAST["message"])


def _recommended(w, h):
    r = pictures.recommended(int(w or 0), int(h or 0))
    size = (r["width"], r["height"]) if r["width"] else (gr.skip(), gr.skip())
    return (r["sampler"], r["scheduler"], r["steps"], r["cfg"]) + size


def _encode_refs(p, eng, prepared):
    """VAE-encode every cropped picture once, on the host's reference channel (engine keeps the latents)."""
    from backend.args import dynamic_args
    from modules.sd_samplers_common import images_tensor_to_samples

    dynamic_args.is_referencing = True
    try:
        for img in prepared:
            t = torch.from_numpy(pictures.to_chw_array(img)).to(device=shared.device)
            with torch.inference_mode():
                images_tensor_to_samples(t, 0, p.sd_model)
    finally:
        dynamic_args.is_referencing = False
    if len(eng.fibo_refs) != len(prepared):
        raise RuntimeError(f"{len(prepared)} pictures but {len(eng.fibo_refs)} reference latents")


class FiboEditPanel(scripts.Script):
    sorting_priority = 540

    def __init__(self):
        super().__init__()
        self.found = {}
        self.button = None
        self.wired = False

    def title(self):
        return "FIBO 2.0 Edit"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    # ---------------------------------------------------------------- UI
    def ui(self, is_img2img):
        with InputAccordion(value=False, label="FIBO 2.0 Edit") as enable:
            gr.Markdown("The img2img picture is picture 1. Add up to 4 more. Write the edit as the prompt, plain text or JSON.")
            with gr.Row():
                extras = [gr.Image(type="pil", label=f"Picture {i + 2}", sources=["upload", "clipboard"], height=160) for i in range(EXTRA_SLOTS)]
            with gr.Row():
                shift_mode = gr.Radio(choices=list(MODES), value=MODE_SIZE, label="Shift")
                manual = gr.Checkbox(value=False, label="Use my own width and height")
            self.button = gr.Button("Recommended settings (Euler, 4 steps, CFG 1, size of picture 1)")
            with gr.Row():
                status = gr.Textbox(value="", label="Status", interactive=False, max_lines=1, scale=5)
                refresh = gr.Button("Check", scale=1)
            gr.Markdown(LICENCE)
        refresh.click(fn=_status, outputs=[status], show_progress=False)
        enable.change(fn=_status, outputs=[status], show_progress=False)
        self._wire()
        return [enable, shift_mode, manual] + extras

    def after_component(self, component, **kwargs):
        eid = kwargs.get("elem_id")
        if eid in TARGETS:
            self.found[eid] = component
            self._wire()

    def _wire(self):
        """UNVERIFIED in a live UI: the button is wired once the host's six controls are known."""
        if self.wired or self.button is None or any(t not in self.found for t in TARGETS):
            return
        try:
            outs = [self.found[t] for t in TARGETS]
            self.button.click(fn=_recommended, _js="currentImg2imgSourceResolution", inputs=outs[4:], outputs=outs, show_progress=False)
            self.wired = True
        except Exception as e:
            print(f"{errors.TAG} recommended-settings button not wired: {e}")

    # ---------------------------------------------------------------- generation
    def process(self, p, enable, shift_mode, manual, *extras):
        hooks.guarded("panel start", self._start, p, bool(enable), shift_mode, bool(manual), list(extras))

    def _start(self, p, enable, shift_mode, manual, extras):
        eng = _engine(p.sd_model)
        if eng is None:
            if enable:
                raise errors.FiboRefusal(errors.NOT_THIS_MODEL)  # printed once by hooks.guarded
            return
        eng.fibo_reset()
        if not enable:
            return  # the engine refuses at text encoding: this model needs the panel
        first = p.init_images[0] if getattr(p, "init_images", None) else None
        mask = getattr(p, "image_mask", None)
        work = job.build_job(first, extras, p.prompt, mask=mask, invert=bool(getattr(p, "inpainting_mask_invert", 0)),
                             manual=manual, typed_w=p.width, typed_h=p.height)
        # spec 11.3: no host mask blend, no crop, no paste-back; the mask is already painted into picture 1
        p.image_mask = None
        p.latent_mask = None
        p.mask_for_overlay = None
        p.inpaint_full_res = False
        p.init_images = [work["pictures"][0]]
        p.width, p.height = work["width"], work["height"]
        p.resize_mode = 0
        p.denoising_strength = 1.0  # spec 9.6: start from pure noise
        p.clear_prompt_cache()  # the same text with other pictures must encode again
        eng.shift_mode = shift_mode if shift_mode in MODES else MODE_SIZE
        eng.fibo_start(work["prompt"], work["vision"])
        _encode_refs(p, eng, work["pictures"])
        if getattr(p, "cfg_scale", 1.0) > 1.0:
            errors.say(f"CFG {p.cfg_scale}: a second pass with the empty caption runs (slower). The model is made for CFG 1.")
        p.extra_generation_params["FIBO Edit"] = f"{len(work['pictures'])} picture(s), shift {eng.shift_mode}"
        errors.say(f"{len(work['pictures'])} picture(s) ready, output {p.width}x{p.height}.")

    def process_before_every_sampling(self, p, *args, **kwargs):
        hooks.guarded("before sampling", self._before_sampling, p)

    def _before_sampling(self, p):
        eng = _engine(p.sd_model)
        if eng is None or not eng.job_active:
            return
        eng.apply_size_shift(p.width, p.height)  # spec 9.4: from the real size, every pass
        p.mask = p.nmask = None
        p.overlay_images = None
        p.paste_to = None
        eng.publish_refs()

    def postprocess(self, p, processed, *args):
        eng = _engine(getattr(p, "sd_model", None))
        if eng is not None:
            hooks.guarded("panel end", eng.fibo_reset)
