# Host wrappers with fake host functions: detection, component build, concat, loader, guard. Author: Stable Yogi.
import os
import tempfile
import types
import unittest

from _common import ROOT  # noqa: F401
from fibo_edit import errors, gguf_names, hooks, weights
from fibo_edit.errors import FiboRefusal
from test_detect import _find_q8
from test_files import fake_safetensors

P = "model.diffusion_model."


class Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)


def fresh():
    hooks.STATE.update(off=False, why="")


class Guard(unittest.TestCase):
    def setUp(self):
        fresh()

    def tearDown(self):
        fresh()

    def test_refusal_passes_through(self):
        def stop():
            raise FiboRefusal("no")
        with self.assertRaises(FiboRefusal):
            hooks.guarded("x", stop)
        self.assertFalse(hooks.is_off())

    def test_error_turns_off_once(self):
        self.assertIsNone(hooks.guarded("stage A", lambda: 1 / 0))
        self.assertTrue(hooks.is_off())
        self.assertIn("stage A", errors.LAST["message"])
        self.assertIsNone(hooks.guarded("later", lambda: 5))  # off: our steps no longer run


class Detect(unittest.TestCase):
    def setUp(self):
        fresh()

    def test_other_models_go_to_host(self):
        calls = []
        det = hooks.make_detect(lambda sd, prefix: calls.append(prefix) or {"image_model": "flux"})
        self.assertEqual(det({P + "x.weight": Shape((1,))}, P), {"image_model": "flux"})
        self.assertEqual(calls, [P])

    def test_broken_fibo_shapes_disable_and_fall_back(self):
        sd = {P + k: Shape((1,)) for k in weights.FINGERPRINT}
        det = hooks.make_detect(lambda sd, prefix: None)
        self.assertIsNone(det(sd, P))
        self.assertTrue(hooks.is_off())
        fresh()

    @unittest.skipUnless(_find_q8(), "Q8 GGUF not found")
    def test_real_header_config(self):
        head = gguf_names.read_header(_find_q8())
        sd = {P + k: Shape(v[0]) for k, v in head["tensors"].items()}
        sd[P + "layers.0.attention.comfy_quant"] = Shape((3,))
        cfg = hooks.make_detect(lambda *a: None)(sd, P)
        self.assertEqual(cfg["image_model"], hooks.IMAGE_MODEL)
        self.assertEqual((cfg["dim"], cfg["n_layers"], cfg["injection_layers"]), (3584, 44, [6, 14, 22, 30, 38]))


class Build(unittest.TestCase):
    def setUp(self):
        fresh()
        self.calls = []
        self.mod = types.SimpleNamespace(SingleStreamDiT="host class")

        def orig(guess, name, lib, cls, repo, sd):
            self.calls.append((name, cls, repo, self.mod.SingleStreamDiT, sorted(sd) if isinstance(sd, dict) else sd))
            if cls == "boom":
                raise ValueError("x")
            return name

        self.lhc = hooks.make_lhc(orig, lambda g: g == "ours", lambda: hooks.swapped(self.mod, "SingleStreamDiT", "our class"), "TE_REPO", "VAE_REPO")

    def test_transformer_built_by_host_branch_with_our_class(self):
        sd = {"a.weight": 1, "a.comfy_quant": 2}
        self.lhc("ours", "transformer", "diffusers", hooks.TRANSFORMER_CLS, "R", sd)
        self.assertEqual(self.calls[-1], ("transformer", hooks.HOST_TRANSFORMER_CLS, "R", "our class", ["a.weight"]))
        self.assertEqual(self.mod.SingleStreamDiT, "host class")

    def test_class_restored_after_error(self):
        lhc = hooks.make_lhc(lambda *a: 1 / 0, lambda g: True, lambda: hooks.swapped(self.mod, "SingleStreamDiT", "our class"), "T", "V")
        with self.assertRaises(ZeroDivisionError):
            lhc("ours", "transformer", "diffusers", hooks.TRANSFORMER_CLS, "R", {})
        self.assertEqual(self.mod.SingleStreamDiT, "host class")

    def test_config_folders_redirected(self):
        for name, want in (("text_encoder", "TE_REPO"), ("tokenizer", "TE_REPO"), ("vae", "VAE_REPO")):
            self.lhc("ours", name, "x", "C", "R", None)
            self.assertEqual(self.calls[-1][2], want)

    def test_other_models_untouched(self):
        self.lhc("theirs", "vae", "x", "C", "R", None)
        self.assertEqual(self.calls[-1][2], "R")


class Concat(unittest.TestCase):
    def test_lengths(self):
        on = {"v": True}
        cc = hooks.make_can_concat(lambda s, o: "host", lambda: on["v"])
        a, b, c = (types.SimpleNamespace(cond=Shape(s)) for s in ((1, 5, 8), (1, 9, 8), (1, 5, 8)))
        self.assertFalse(cc(a, b))
        self.assertEqual(cc(a, c), "host")
        on["v"] = False
        self.assertEqual(cc(a, b), "host")


class Loader(unittest.TestCase):
    def setUp(self):
        fresh()
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.bare = fake_safetensors(os.path.join(t, "fibo2_edit.safetensors"), list(weights.FINGERPRINT))
        self.other = fake_safetensors(os.path.join(t, "sdxl.safetensors"), ["x"])
        self.te_dir, self.vae_dir = os.path.join(t, "te"), os.path.join(t, "vae")
        self.got, self.loads = [], []

    def tearDown(self):
        self.tmp.cleanup()
        fresh()

    def loader(self):
        def orig(sd, additional_state_dicts=None):
            self.got.append(list(additional_state_dicts or []))
            return "engine"
        return hooks.make_loader(orig, lambda: self.loads.append(1), lambda: [self.te_dir], lambda: [self.vae_dir])

    def test_bare_file_gets_modules(self):
        te = fake_safetensors(os.path.join(self.te_dir, "qwen3vl_4b.safetensors"), ["a"])
        vae = fake_safetensors(os.path.join(self.vae_dir, "flux2-vae.safetensors"), ["b"])
        self.assertEqual(self.loader()(self.bare, ["lora_like.safetensors"]), "engine")
        self.assertEqual(self.got[-1], [te, vae, "lora_like.safetensors"])
        self.assertEqual(self.loads, [1])

    def test_missing_module_refused_with_folder(self):
        fake_safetensors(os.path.join(self.vae_dir, "flux2-vae.safetensors"), ["b"])
        with self.assertRaises(FiboRefusal) as c:
            self.loader()(self.bare)
        self.assertIn(self.te_dir, str(c.exception))
        self.assertEqual(self.got, [])

    def test_other_model_untouched(self):
        self.loader()(self.other, None)
        self.assertEqual(self.got[-1], [])
        self.assertEqual(self.loads, [1])


if __name__ == "__main__":
    unittest.main()
