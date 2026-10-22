# Text pass with a fake Qwen3-VL shaped encoder: splice, positions, deepstack, taps. CPU. Author: Stable Yogi.
import unittest

import torch
from torch import nn

from _common import ROOT  # noqa: F401
from fibo_edit import host_text
from fibo_edit.taps import TAP_LAYERS

IMG = host_text.IMAGE_TOKEN
D = 4


class AddOne(nn.Module):
    def forward(self, x):
        return x + 1, None  # the host's blocks return (x, kv)


class FakeTE(nn.Module):
    """Duck-types the host encoder: embeddings, vision pre-pass, image inputs, 36 blocks, taps."""

    def __init__(self, deepstack=3):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(IMG + 1, D)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([AddOne() for _ in range(36)])
        self.n_deep = deepstack
        self.seen = {}

    def get_input_embeddings(self):
        return self.embed

    def preprocess_embed(self, item, device=None):
        data = item["data"]
        n = int(data.shape[1] * data.shape[2] // 4)  # toy: 4 pixels per vision token
        return torch.full((n, D), 100.0 + float(data.mean())), {"grid": (1, data.shape[1], data.shape[2]), "deepstack": [torch.ones(n, D) * 10 for _ in range(self.n_deep)]}

    def build_image_inputs(self, embeds, info):
        L = embeds.shape[1]
        mask = torch.zeros((1, L), dtype=torch.bool)
        deep = [[] for _ in range(self.n_deep)]
        for e in info:
            mask[0, e["index"]:e["index"] + e["size"]] = True
            for i, f in enumerate(e["extra"]["deepstack"]):
                deep[i].append(f)
        pos = torch.arange(L)[None].repeat(3, 1)
        return pos, mask, [torch.cat(d) for d in deep]

    def forward(self, x, embeds=None, attention_mask=None, num_tokens=None, embeds_info=None, intermediate_output=None,
                final_layer_norm_intermediate=True, position_ids=None, **kw):
        self.seen = dict(position_ids=position_ids, num_tokens=num_tokens, final=final_layer_norm_intermediate)
        h, taps = embeds, []
        for i, layer in enumerate(self.model.layers):
            if i in intermediate_output:
                taps.append(h)
            h = layer(h)[0]
        return h, torch.stack(taps, dim=1)


class Tok:
    def __init__(self, n_extra=0):
        self.n = n_extra

    def __call__(self, texts):
        ids = []
        for part in texts[0].split("<img>"):
            ids += [1 + (ord(c) % 50) for c in part] + [IMG]
        return {"input_ids": [ids[:-1] + [7] * self.n]}


def pic(h, w, v=0.5):
    return torch.full((1, h, w, 3), v)


class Splice(unittest.TestCase):
    def test_segments(self):
        self.assertEqual(host_text.segments([1, 2, IMG, 3, IMG]), [[1, 2], [3], []])
        self.assertEqual(host_text.segments([5]), [[5]])

    def test_splice_layout(self):
        te = FakeTE()
        emb, info = host_text.splice(te, [1, 2, IMG, 3, IMG], [pic(2, 6), pic(2, 4, 0.0)], "cpu")
        self.assertEqual(emb.shape, (1, 3 + 3 + 2, D))
        self.assertEqual([(e["index"], e["size"]) for e in info], [(2, 3), (6, 2)])
        table = te.embed.weight.detach()
        self.assertTrue(torch.equal(emb[0, 0], table[1]) and torch.equal(emb[0, 1], table[2]) and torch.equal(emb[0, 5], table[3]))
        self.assertTrue((emb[0, 2:5] == 100.5).all() and (emb[0, 6:8] == 100.0).all())
        self.assertEqual(emb.dtype, torch.float32)

    def test_count_mismatch(self):
        with self.assertRaises(ValueError):
            host_text.splice(FakeTE(), [1, IMG, 2], [], "cpu")

    def test_truncation(self):
        ids = host_text.token_ids(Tok(n_extra=5000), "abc")
        self.assertEqual(len(ids), host_text.MAX_TOKENS)


class Encoder(unittest.TestCase):
    def test_taps_positions_and_deepstack(self):
        te = FakeTE(deepstack=3)
        emb, info = host_text.splice(te, [1, IMG, 2], [pic(2, 2)], "cpu")
        taps = host_text.run_encoder(te, emb, info)
        self.assertEqual(taps.shape, (len(TAP_LAYERS), 3, D))
        self.assertEqual(tuple(te.seen["position_ids"].shape), (3, 3))
        self.assertEqual(te.seen["num_tokens"], [3])
        self.assertFalse(te.seen["final"])
        # first tap = state before block 5: text +5, the picture token +5 and +10 from each of 3 deepstack blocks
        self.assertTrue(torch.allclose(taps[0, 1], emb[0, 1] + 5 + 30))
        self.assertTrue(torch.allclose(taps[0, 0], emb[0, 0] + 5))
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in te.model.layers))

    def test_text_only_has_no_vision_inputs(self):
        te = FakeTE()
        emb, info = host_text.splice(te, [1, 2, 3], [], "cpu")
        host_text.run_encoder(te, emb, info)
        self.assertIsNone(te.seen["position_ids"])

    def test_hooks_removed_on_error(self):
        te = FakeTE()
        emb, info = host_text.splice(te, [1, IMG], [pic(2, 2)], "cpu")
        te.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        with self.assertRaises(RuntimeError):
            host_text.run_encoder(te, emb, info)
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in te.model.layers))

    def test_encode_context(self):
        te = FakeTE()
        ctx = host_text.encode(te, Tok(), "ab<img>c", [pic(2, 4)], "cpu")
        self.assertEqual(ctx.shape, (2 + 2 + 1, 18 * D))
        self.assertEqual(ctx.dtype, torch.bfloat16)

    def test_missing_taps_refused(self):
        te = FakeTE()
        te.forward = lambda *a, **k: (None, None)
        emb, info = host_text.splice(te, [1, 2], [], "cpu")
        with self.assertRaises(RuntimeError):
            host_text.run_encoder(te, emb, info)


if __name__ == "__main__":
    unittest.main()
