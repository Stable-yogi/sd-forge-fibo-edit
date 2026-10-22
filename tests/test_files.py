# Bare-file check and the text encoder / VAE pick. Fake header-only files plus the real Q8 header. Author: Stable Yogi.
import json
import os
import struct
import tempfile
import unittest

from _common import ROOT  # noqa: F401
from fibo_edit import files, weights
from test_detect import _find_q8


def fake_safetensors(path, keys):
    head = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    raw = json.dumps(head).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return path


class Pick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.te_dir = os.path.join(self.tmp.name, "text_encoder")
        self.vae_dir = os.path.join(self.tmp.name, "VAE")
        os.makedirs(self.te_dir)
        os.makedirs(self.vae_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_by_name(self):
        te = fake_safetensors(os.path.join(self.te_dir, "qwen3vl_4b_fp8_scaled.safetensors"), ["a"])
        vae = fake_safetensors(os.path.join(self.vae_dir, "flux2-vae.safetensors"), ["b"])
        plan = files.plan_modules([], [self.te_dir], [self.vae_dir])
        self.assertEqual(plan, {"add": [te, vae], "missing": []})

    def test_by_key_in_sub_folder(self):
        te = fake_safetensors(os.path.join(self.te_dir, "sub", "encoder.safetensors"), [files.TE_MARK])
        vae = fake_safetensors(os.path.join(self.vae_dir, "deep", "ae.safetensors"), ["bn.running_mean", "x"])
        fake_safetensors(os.path.join(self.vae_dir, "other.safetensors"), ["z"])
        self.assertEqual(files.find_text_encoder([self.te_dir]), te)
        self.assertEqual(files.find_vae([self.vae_dir]), vae)

    def test_missing_names_the_folders(self):
        plan = files.plan_modules([], [self.te_dir, os.path.join(self.tmp.name, "nope")], [self.vae_dir])
        self.assertEqual(plan["add"], [])
        self.assertEqual([m[0] for m in plan["missing"]], ["Qwen3-VL 4B text encoder", "FLUX.2 VAE"])
        self.assertIn(self.te_dir, plan["missing"][0][1])

    def test_user_choice_wins(self):
        fake_safetensors(os.path.join(self.te_dir, "qwen3vl_4b.safetensors"), ["a"])
        mine = fake_safetensors(os.path.join(self.tmp.name, "my_flux2_vae.safetensors"), ["c"])
        plan = files.plan_modules([mine], [self.te_dir], [self.vae_dir])
        self.assertEqual(len(plan["add"]), 1)
        self.assertEqual(plan["missing"], [])

    def test_foreign_vae_is_found(self):
        wan = fake_safetensors(os.path.join(self.tmp.name, "other_vae.safetensors"), ["decoder.conv1.weight", "encoder.conv1.weight"])
        ok = fake_safetensors(os.path.join(self.tmp.name, "flux2-vae.safetensors"), ["decoder.conv_in.weight", "encoder.conv_in.weight", "bn.running_mean"])
        self.assertEqual(files.foreign_vae([wan]), wan)
        self.assertIsNone(files.foreign_vae([ok]))
        self.assertIsNone(files.foreign_vae([]))

    def test_bare_fibo_safetensors(self):
        bare = fake_safetensors(os.path.join(self.tmp.name, "fibo.safetensors"), list(weights.FINGERPRINT))
        full = fake_safetensors(os.path.join(self.tmp.name, "aio.safetensors"), list(weights.FINGERPRINT) + ["text_encoders.x"])
        other = fake_safetensors(os.path.join(self.tmp.name, "sdxl.safetensors"), ["model.diffusion_model.input_blocks.0.0.weight"])
        self.assertTrue(files.is_bare_fibo(bare))
        self.assertFalse(files.is_bare_fibo(full))
        self.assertFalse(files.is_bare_fibo(other))

    def test_never_raises(self):
        junk = os.path.join(self.tmp.name, "junk.safetensors")
        with open(junk, "wb") as f:
            f.write(b"not a header")
        self.assertFalse(files.is_bare_fibo(junk))
        self.assertFalse(files.is_bare_fibo(os.path.join(self.tmp.name, "absent.gguf")))
        self.assertFalse(files.is_bare_fibo(os.path.join(self.tmp.name, "model.ckpt")))


@unittest.skipUnless(_find_q8(), "Q8 GGUF not found")
class RealFile(unittest.TestCase):
    def test_q8_is_bare(self):
        self.assertTrue(files.is_bare_fibo(_find_q8()))


if __name__ == "__main__":
    unittest.main()
