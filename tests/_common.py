# Shared test helpers. Author: Stable Yogi.
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIX = os.environ.get("FIBO_FIXTURES", os.path.join(ROOT, "tests", "fixtures"))


def fixtures() -> dict:
    if not os.path.isfile(os.path.join(FIX, "fixtures.json")):
        raise unittest.SkipTest("reference data not present (set FIBO_FIXTURES to its folder)")
    with open(os.path.join(FIX, "fixtures.json"), encoding="utf-8") as f:
        return json.load(f)


def npy(name: str):
    import numpy as np

    return np.load(os.path.join(FIX, name + ".npy"))


def tensor(name: str):
    import torch

    return torch.from_numpy(npy(name).copy())
