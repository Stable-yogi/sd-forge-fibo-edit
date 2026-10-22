# Pictures, job building, panel status and the repo config. Pure, CPU. Author: Stable Yogi.
import json
import os
import unittest

import numpy as np
from PIL import Image

from _common import ROOT
from fibo_edit import errors, job, pictures
from fibo_edit.errors import FiboRefusal


def img(w, h, colour=(200, 30, 30)):
    return Image.new("RGB", (w, h), colour)


class Prepare(unittest.TestCase):
    def test_trained_sizes(self):
        for (w, h), want in {(4000, 3000): (1152, 864), (1920, 1080): (1424, 800), (512, 512): (1024, 1024), (300, 600): (720, 1440)}.items():
            self.assertEqual(pictures.prepare(img(w, h)).size, want, (w, h))

    def test_rgb_and_crop_centre(self):
        a = Image.new("RGBA", (2048, 1024), (0, 0, 255, 255))
        out = pictures.prepare(a)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, pictures.trained_size_for(2048, 1024))

    def test_vision_copy(self):
        self.assertEqual(pictures.vision_copy(img(1152, 864)).size, (768, 576))
        self.assertEqual(pictures.vision_copy(img(500, 400)).size, (500, 400))  # never enlarged
        self.assertEqual(pictures.to_unit_array(img(8, 4)).shape, (1, 4, 8, 3))
        chw = pictures.to_chw_array(img(8, 4, (255, 0, 0)))
        self.assertEqual(chw.shape, (1, 3, 4, 8))
        self.assertAlmostEqual(float(chw[0, 0].max()), 1.0)

    def test_sizes(self):
        self.assertEqual(pictures.snap_size(1000, 30), (992, 64))
        self.assertEqual(pictures.output_size(img(1600, 900), True, 1000, 1000), (992, 992))
        self.assertEqual(pictures.output_size(img(1600, 900), False, 1000, 1000), (1424, 800))
        r = pictures.recommended(1600, 900)
        self.assertEqual((r["sampler"], r["steps"], r["cfg"], r["width"], r["height"]), ("Euler", 4, 1.0, 1424, 800))
        self.assertIsNone(pictures.recommended(0, 0)["width"])


class Checks(unittest.TestCase):
    def test_counts(self):
        with self.assertRaises(FiboRefusal) as c:
            pictures.check_count(0)
        self.assertEqual(str(c.exception), errors.NO_PICTURE)
        with self.assertRaises(FiboRefusal) as c:
            pictures.check_count(6)
        self.assertIn("6", str(c.exception))
        pictures.check_count(5)

    def test_prompt_and_mask(self):
        with self.assertRaises(FiboRefusal):
            pictures.check_prompt(" \n\t ")
        with self.assertRaises(FiboRefusal) as c:
            pictures.check_mask(2, (8, 8), (8, 8))
        self.assertEqual(str(c.exception), errors.MASK_MANY)
        with self.assertRaises(FiboRefusal) as c:
            pictures.check_mask(1, (8, 9), (8, 8))
        self.assertIn("8x9", str(c.exception))
        pictures.check_mask(1, None, (8, 8))

    def test_apply_mask_threshold_and_invert(self):
        m = np.zeros((2, 2), np.uint8)
        m[0, 0], m[0, 1] = 128, 127
        out = np.asarray(pictures.apply_mask(img(2, 2), Image.fromarray(m)))
        self.assertEqual(tuple(out[0, 0]), (128, 128, 128))
        self.assertEqual(tuple(out[0, 1]), (200, 30, 30))
        inv = np.asarray(pictures.apply_mask(img(2, 2), Image.fromarray(m), invert=True))
        self.assertEqual(tuple(inv[0, 0]), (200, 30, 30))
        self.assertEqual(tuple(inv[0, 1]), (128, 128, 128))


class BuildJob(unittest.TestCase):
    def test_one_picture(self):
        w = job.build_job(img(400, 300), [None, None], "make it blue")
        self.assertEqual((w["width"], w["height"]), (1152, 864))
        self.assertEqual(w["pictures"][0].size, (1152, 864))
        self.assertEqual(w["vision"][0].shape, (1, 576, 768, 3))
        self.assertEqual(w["prompt"], "make it blue")

    def test_extra_pictures_keep_their_own_size(self):
        w = job.build_job(img(400, 300), [None, img(300, 600), img(512, 512)], ["swap them"])
        self.assertEqual([p.size for p in w["pictures"]], [(1152, 864), (720, 1440), (1024, 1024)])
        self.assertEqual(w["prompt"], "swap them")

    def test_mask_painted_before_resize(self):
        w = job.build_job(img(64, 64, (0, 0, 0)), [], "fill", mask=Image.new("L", (64, 64), 255))
        arr = np.asarray(w["pictures"][0])
        self.assertTrue((arr == 128).all())

    def test_refusals(self):
        cases = [
            (dict(first=None, extras=[], prompt="x"), errors.NO_PICTURE),
            (dict(first=img(64, 64), extras=[], prompt="   "), errors.EMPTY_PROMPT),
            (dict(first=img(64, 64), extras=[img(64, 64)], prompt="x", mask=Image.new("L", (64, 64))), errors.MASK_MANY),
            (dict(first=img(64, 64), extras=[img(64, 64)] * 5, prompt="x"), errors.TOO_MANY.format(n=6)),
        ]
        for kw, msg in cases:
            with self.assertRaises(FiboRefusal) as c:
                job.build_job(**kw)
            self.assertEqual(str(c.exception), msg)
        with self.assertRaises(FiboRefusal) as c:
            job.build_job(img(64, 64), [], "x", mask=Image.new("L", (32, 64)))
        self.assertIn("32x64", str(c.exception))

    def test_manual_size(self):
        w = job.build_job(img(400, 300), [], "x", manual=True, typed_w=777, typed_h=500)
        self.assertEqual((w["width"], w["height"]), (768, 496))

    def test_host_mask_alpha(self):
        rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        rgba.putpixel((1, 1), (0, 0, 0, 255))
        grey = job.mask_from_host(rgba)
        self.assertEqual(grey.getpixel((1, 1)), 255)
        self.assertEqual(grey.getpixel((0, 0)), 0)
        self.assertIsNone(job.mask_from_host(None))


class Status(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(job.status_text(False, False), errors.INERT)
        self.assertEqual(job.status_text(True, False), job.READY)
        self.assertEqual(job.status_text(True, False, last="2 picture(s) ready"), "2 picture(s) ready")
        self.assertIn("boom", job.status_text(True, True, why="boom"))

    def test_say_and_refuse(self):
        with self.assertRaises(FiboRefusal):
            errors.refuse("stop here")
        self.assertEqual(errors.LAST["message"], "stop here")


class RepoConfig(unittest.TestCase):
    def test_model_index(self):
        with open(os.path.join(ROOT, "fibo_edit", "repo", "model_index.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["transformer"], ["diffusers", "FiboEditTransformer2DModel"])
        self.assertEqual(cfg["text_encoder"], ["transformers", "Qwen3VLModel"])
        self.assertEqual(cfg["tokenizer"], ["transformers", "Qwen2Tokenizer"])
        self.assertEqual(cfg["vae"], ["diffusers", "AutoencoderKLFlux2"])
        self.assertNotIn("scheduler", cfg)


if __name__ == "__main__":
    unittest.main()
