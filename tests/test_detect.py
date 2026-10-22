# T3: detection on the real Q8 GGUF header (names and shapes only) + name rules. Author: Stable Yogi.
import glob
import os
import unittest

from _common import ROOT  # noqa: F401  (puts the package on sys.path)
from fibo_edit import gguf_names, weights

MODEL_DIR = os.environ.get("FIBO_MODEL_DIR", r"F:\Models\DiffusionModels")


def _find_q8():
    hits = glob.glob(os.path.join(MODEL_DIR, "*", "*Q8_0*.gguf"))
    hits = [h for h in hits if "fibo" in os.path.basename(h).lower()]
    return hits[0] if hits else None


class NameRules(unittest.TestCase):
    def test_fingerprint_needs_all_three(self):
        names = list(weights.FINGERPRINT)
        self.assertTrue(weights.is_fibo(names))
        for i in range(3):
            self.assertFalse(weights.is_fibo(names[:i] + names[i + 1:]))

    def test_prefix_found(self):
        names = ["model.diffusion_model." + n for n in weights.FINGERPRINT]
        self.assertEqual(weights.find_prefix(names), "model.diffusion_model.")

    def test_marker_keys_ignored(self):
        st = weights.strip({"a.weight": 1, "a.comfy_quant": 2}, "")
        self.assertEqual(list(st), ["a.weight"])


@unittest.skipUnless(_find_q8(), "Q8 GGUF not found")
class RealHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head = gguf_names.read_header(_find_q8())
        cls.shapes = {k: v[0] for k, v in cls.head["tensors"].items()}

    def test_634_tensors_and_fingerprint(self):
        self.assertEqual(len(self.shapes), 634)
        self.assertTrue(weights.is_fibo(self.shapes))

    def test_architecture_field_not_used(self):
        arch = self.head["metadata"].get("general.architecture")
        self.assertTrue(weights.is_fibo(self.shapes), f"arch field says {arch!r}; detection ignores it")

    def test_config_matches_spec(self):
        cfg = weights.config_from_shapes(self.shapes)
        self.assertEqual(cfg["dim"], 3584)
        self.assertEqual(cfg["in_channels"], 128)
        self.assertEqual(cfg["out_channels"], 128)
        self.assertEqual(cfg["n_layers"], 44)
        self.assertEqual(cfg["n_refiner_layers"], 2)
        self.assertEqual(cfg["injection_layers"], [6, 14, 22, 30, 38])
        self.assertEqual(cfg["text_bundle_width"], 7680)
        self.assertEqual(cfg["ffn_dim"], 9728)
        self.assertEqual(cfg["max_summary_tokens"], 512)
        self.assertEqual(cfg["perceiver_layers"], 2)

    def test_every_name_has_a_home_in_the_model(self):
        import torch

        from fibo_edit.dit import FiboEditDiT

        with torch.device("meta"):
            model = FiboEditDiT(**weights.config_from_shapes(self.shapes))
        mine = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        theirs = {k: tuple(v) for k, v in self.shapes.items() if not weights.is_ignored(k)}
        self.assertEqual(sorted(set(theirs) - set(mine)), [])
        self.assertEqual(sorted(set(mine) - set(theirs)), [])
        bad = [(k, theirs[k], mine[k]) for k in theirs if theirs[k] != mine[k] and not (theirs[k] == () and mine[k] == (1,))]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
