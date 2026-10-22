# T1: DiT forward on the tiny config, fixtures F1..F5, max abs error 1e-4. CPU only. Author: Stable Yogi.
import os
import unittest

import torch

from _common import FIX, fixtures, tensor
from fibo_edit import weights

FX = fixtures()
TOL = 1e-4


def _model(split=False):
    state = weights.load_npz(os.path.join(FIX, "tiny_dit_weights.npz"))
    return weights.build_from_state(state, split_ffn=split).eval()


def _run(model, case):
    c = FX["dit_cases"][case]
    refs = [tensor(r) for r in c["refs"]]
    with torch.no_grad():
        return model.run(tensor(c["x"]), tensor(c["t"]), tensor(c["ctx"]), refs)


class TinyDiT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = _model()

    def _check(self, case):
        c = FX["dit_cases"][case]
        out = _run(self.model, case)
        want = tensor(c["out"])
        self.assertEqual(out.shape, want.shape)
        err = float((out - want).abs().max())
        self.assertLessEqual(err, TOL, f"{case} max abs err {err:.3e}")
        return out

    def test_config_from_names(self):
        cfg = weights.config_from_shapes(weights.shapes_of(weights.load_npz(os.path.join(FIX, "tiny_dit_weights.npz"))))
        t = FX["tiny_config"]
        self.assertEqual(cfg["dim"], t["dim"])
        self.assertEqual(cfg["injection_layers"], t["injection_layers"])
        self.assertEqual(cfg["text_bundle_width"], t["text_bundle_width"])
        self.assertEqual(cfg["ffn_dim"], t["ffn_dim"])
        self.assertEqual(cfg["n_layers"], t["n_layers"])

    def test_f1_no_reference(self):
        self._check("F1")

    def test_f2_one_reference_and_differs(self):
        out2 = self._check("F2")
        out1 = _run(self.model, "F1")
        self.assertGreater(float((out2 - out1).abs().max()), 1e-3)

    def test_f3_batch_two_refs_two_sizes(self):
        self._check("F3")

    def test_f4_split_ffn_equal(self):
        split = _model(split=True)
        a, b = _run(self.model, "F2"), _run(split, "F2")
        self.assertEqual(float((a - b).abs().max()), FX["dit_cases"]["F4"]["max_abs_diff_vs_F2"])

    def test_f5_long_text_384_summary(self):
        from fibo_edit.sizes import summary_count

        self.assertEqual(summary_count(tensor("F5_ctx").shape[1]), FX["dit_cases"]["F5"]["num_summary"])
        self._check("F5")


if __name__ == "__main__":
    unittest.main()
