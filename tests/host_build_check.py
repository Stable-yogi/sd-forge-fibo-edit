# Author: Stable Yogi. Dev check, needs a Forge Neo install; not part of run_all.py. Run with the host venv python -I.
# CPU-only: build the tiny FIBO DiT through the REAL host Krea2 branch (forge ops, non-strict load) with our class
# swapped in, run F1..F5 through the host-style forward and the host attention kernel, compare to the fixtures.
import os, sys, runpy
HOST = os.environ.get("FORGE_HOME", r"L:\Forge Neo\sd-webui-forge-neo")
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0], "--cpu"]
for p in (HOST, os.path.join(HOST, "modules_forge", "packages"), EXT, os.path.join(EXT, "tests")):
    sys.path.insert(0, p)
os.chdir(HOST)
import torch
import backend.loader as loader
from backend.args import dynamic_args
from fibo_edit import hooks, weights, dit
reg = runpy.run_path(os.path.join(EXT, "scripts", "fibo_register.py"), run_name="fibo_register_check")
hooks.STATE.update(off=False, why="")
from _common import FIX, fixtures, tensor
FX = fixtures()
state = weights.load_npz(os.path.join(FIX, "tiny_dit_weights.npz"))
state = {k: v.float() for k, v in state.items()}
Base = reg["FiboEditBase"] if "FiboEditBase" in reg else __import__("huggingface_guess.model_list", fromlist=["x"]).FiboEditBase
g = Base.__new__(Base)
g.unet_config = weights.config_from_shapes(weights.shapes_of(state))
g.nunchaku = False
g.supported_inference_dtypes = [torch.float32]
dynamic_args.forge_unet_storage_dtype = torch.float32
model = loader.load_huggingface_component(g, "transformer", "diffusers", hooks.TRANSFORMER_CLS, Base.huggingface_repo, dict(state))
print("built:", type(model).__name__, "storage", model.storage_dtype, "compute", model.computation_dtype, "attn:", dit.ATTEND["fn"].__name__)
sd_back = model.state_dict()
missing = [k for k in state if k not in sd_back]
diff = max(float((sd_back[k].float() - state[k]).abs().max()) for k in state if k in sd_back)
print("keys missing after load:", missing[:5], "max weight diff:", diff)
worst = 0.0
for case in ("F1", "F2", "F3", "F4", "F5"):
    c = FX["dit_cases"].get(case)
    if c is None or "x" not in c:
        print(case, "not in fixtures"); continue
    dynamic_args.ref_latents = [tensor(r) for r in c.get("refs", [])]
    with torch.no_grad():
        out = model.forward(tensor(c["x"]), tensor(c["t"]), tensor(c["ctx"]))
    err = float((out - tensor(c["out"])).abs().max()); worst = max(worst, err)
    print(case, tuple(out.shape), "max abs err %.3e" % err)
print("PASS" if worst <= 1e-4 and not missing else "FAIL")
