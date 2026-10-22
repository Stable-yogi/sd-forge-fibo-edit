# T2: pure helpers against fixtures F6..F16. CPU only. Author: Stable Yogi.
import json
import unittest
from fractions import Fraction

import numpy as np
import torch

from _common import fixtures, npy, tensor
from fibo_edit import latents, mask, prompt, rope, schedule, sizes, taps
from fibo_edit.dit import time_sinus
from fibo_edit.errors import FiboRefusal

FX = fixtures()


class SummaryCount(unittest.TestCase):
    def test_f6(self):
        for length, want in FX["F6_summary_count"].items():
            self.assertEqual(sizes.summary_count(int(length)), want, length)


class Schedules(unittest.TestCase):
    def test_f7_size_based_fixed_rational(self):
        for case in FX["F7_schedules"]:
            w, h, n = case["width"], case["height"], case["steps"]
            self.assertEqual(schedule.latent_tokens(w, h), case["tokens"])
            self.assertAlmostEqual(schedule.mu_for(w, h), case["mu"], places=12)
            np.testing.assert_allclose(schedule.base_sigmas(n)[:-1], case["linear_sigmas"], atol=1e-12)
            np.testing.assert_allclose(schedule.sigmas(w, h, n), case["sigmas_exp_mu"], atol=1e-12)
            np.testing.assert_allclose(schedule.sigmas(w, h, n, schedule.MODE_FIXED), case["sigmas_exp_fixed_1p15"], atol=1e-12)
            rat = [schedule.rational_shift(s, 1.15) for s in schedule.base_sigmas(n)]
            np.testing.assert_allclose(rat, case["sigmas_rational_1p15"], atol=1e-12)

    def test_f8_euler(self):
        sig = FX["F8"]["sigmas_exp_fixed_1p15"]
        x, v = tensor("F8_x"), tensor("F8_v")
        torch.testing.assert_close(schedule.euler_step(x, v, sig[0], sig[1]), tensor("F8_x_next_step0"), atol=1e-6, rtol=0)
        torch.testing.assert_close(schedule.euler_step(x, v, sig[3], sig[4]), tensor("F8_x_next_step3"), atol=1e-6, rtol=0)


class Latents(unittest.TestCase):
    def test_f9_pack_tokens_roundtrip(self):
        raw = tensor("F9_latent_unpacked")
        packed = latents.pack(raw)
        self.assertTrue(torch.equal(packed, tensor("F9_patchified")))
        self.assertTrue(torch.equal(latents.to_tokens(packed), tensor("F9_tokens")))
        self.assertTrue(torch.equal(latents.unpack(packed), raw))
        self.assertTrue(torch.equal(latents.from_tokens(latents.to_tokens(packed), 2, 2), packed))

    def test_f16_normalise(self):
        raw, mean, var = tensor("F16_raw"), tensor("F16_mean"), tensor("F16_var")
        eps = FX["F16"]["eps"]
        out = latents.normalize(raw, mean, var, eps)
        torch.testing.assert_close(out, tensor("F16_normalised"), atol=1e-6, rtol=0)
        back = latents.denormalize(out, mean, var, eps)
        self.assertLessEqual(float((back - raw).abs().max()), 4 * FX["F16"]["max_roundtrip_err"] + 1e-7)


class Rope(unittest.TestCase):
    def test_f10_table(self):
        ids = tensor("F10_rope_ids")
        pe = rope.table(ids)
        self.assertEqual(list(pe.shape), FX["F10_rope_table_shape"])
        torch.testing.assert_close(pe, tensor("F10_rope_table"), atol=1e-5, rtol=0)

    def test_f10_id_lists(self):
        f = FX["F10"]
        self.assertEqual(rope.stream_ids(4, (2, 3), []).int().tolist(), f["no_ref"])
        self.assertEqual(rope.stream_ids(4, (2, 3), [(2, 2)]).int().tolist(), f["one_ref"])
        self.assertEqual(rope.stream_ids(4, (2, 3), [(2, 2), (3, 1)]).int().tolist(), f["two_refs"])

    def test_rotation_keeps_norm(self):
        x = torch.randn(1, 2, 3, 128, generator=torch.Generator().manual_seed(0))
        pe = rope.table(tensor("F10_rope_ids"))
        y = rope.apply(x, pe)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=0)


class TimeSinus(unittest.TestCase):
    def test_f11(self):
        out = time_sinus(tensor("F11_tembed_in"))
        torch.testing.assert_close(out, tensor("F11_tembed_sinusoid"), atol=1e-4, rtol=0)


class Taps(unittest.TestCase):
    def test_f12_layout(self):
        self.assertEqual(list(taps.TAP_LAYERS), FX["F12"]["used_layers_sorted"])
        self.assertEqual(taps.tap_order(), FX["F12"]["index_into_tap"])

    def test_f12_values(self):
        hidden = tensor("F12_hidden_tapped")
        out = taps.build_context(hidden, out_dtype=torch.float32)
        torch.testing.assert_close(out, tensor("F12_out"), atol=1e-5, rtol=0)
        bf = taps.build_context(hidden)
        self.assertEqual(bf.dtype, torch.bfloat16)
        torch.testing.assert_close(bf.float(), out, atol=2e-2, rtol=1e-2)


class Mask(unittest.TestCase):
    def test_f13(self):
        out = mask.paint_grey(npy("F13_image"), mask.mask_to_grey_array(npy("F13_mask")))
        np.testing.assert_array_equal(out, npy("F13_out"))

    def test_threshold_127_stays(self):
        img = np.full((1, 2, 3), 7, np.uint8)
        out = mask.paint_grey(img, np.array([[127, 128]], np.uint8))
        self.assertEqual(out[0, 0].tolist(), [7, 7, 7])
        self.assertEqual(out[0, 1].tolist(), [128, 128, 128])

    def test_size_mismatch_refused(self):
        with self.assertRaises(FiboRefusal):
            mask.paint_grey(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5), np.uint8))


def _vision(s: str) -> str:
    return s.replace("{VISION}", prompt.VISION)


class Prompts(unittest.TestCase):
    def test_f14_edit_wrap(self):
        for case in FX["F14_edit_wrap"]:
            ins, cap = prompt.parse_prompt(case["prompt"])
            self.assertEqual(prompt.wrap(ins, cap, case["num_images"]), _vision(case["wrapped"]), case["prompt"])
            self.assertEqual(prompt.build_text(case["prompt"], case["num_images"]), _vision(case["template_applied"]))

    def test_f14_clean(self):
        for case in FX["F14_generation_clean"]:
            cleaned = prompt.clean_caption(json.loads(case["prompt"]))
            self.assertEqual(prompt.compact(cleaned), case["cleaned"], case["prompt"])

    def test_f14_empty_caption_turn(self):
        self.assertEqual(prompt.empty_caption_turn(), FX["F14_empty_caption_turn"])

    def test_nan_and_nested_lists(self):
        cap = {"a": float("nan"), "b": [[], {"c": None}, "x"], "d": {"preference_score": 1}}
        self.assertEqual(prompt.clean_caption(cap), {"b": ["x"]})

    def test_non_ascii_kept(self):
        ins, cap = prompt.parse_prompt('{"edit_instruction":"Mach den Himmel grün"}')
        self.assertIn("grün", prompt.wrap(ins, cap, 1))

    def test_blank(self):
        self.assertTrue(prompt.is_blank("  \n\t"))
        self.assertFalse(prompt.is_blank(" a "))


class Sizes(unittest.TestCase):
    def test_f15(self):
        for ratio, want in FX["F15_image_size"].items():
            self.assertEqual(list(sizes.trained_size(float(Fraction(ratio)))), want, ratio)

    def test_multiple_of_16(self):
        for a in (0.3, 0.77, 1.0, 1.9, 3.2):
            w, h = sizes.trained_size(a)
            self.assertEqual((w % 16, h % 16), (0, 0))

    def test_cover_crop(self):
        (nw, nh), box = sizes.cover_crop(4000, 3000, 1152, 864)
        self.assertEqual((nw, nh), (1152, 864))
        self.assertEqual(box, (0, 0, 1152, 864))
        (nw, nh), box = sizes.cover_crop(1000, 1000, 1152, 864)
        self.assertEqual((nw, nh), (1152, 1152))
        self.assertEqual(box, (0, 144, 1152, 1008))

    def test_vision(self):
        self.assertEqual(sizes.vision_size(1024, 1024), (768, 768))
        self.assertEqual(sizes.vision_size(500, 300), (500, 300))
        self.assertEqual(sizes.vision_tokens(1024, 1024), 576)
        w, h = sizes.vision_grid(*sizes.vision_size(1424, 800))
        self.assertEqual((w % 32, h % 32), (0, 0))


if __name__ == "__main__":
    unittest.main()
