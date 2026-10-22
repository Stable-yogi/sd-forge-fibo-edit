# Author: Stable Yogi. Dev check, needs a Forge Neo install; not part of run_all.py. Run with the host venv python -I.
# CPU-only: register the architecture into the real host modules, then let the host guess from the real Q8 header.
import os, sys, runpy, traceback
HOST = os.environ.get("FORGE_HOME", r"L:\Forge Neo\sd-webui-forge-neo")
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.argv = [sys.argv[0], "--cpu"]
for p in (HOST, os.path.join(HOST, "modules_forge", "packages"), EXT, os.path.join(EXT, "tests")):
    sys.path.insert(0, p)
os.chdir(HOST)
import backend.loader as loader
import huggingface_guess
from fibo_edit import gguf_names, hooks
reg = runpy.run_path(os.path.join(EXT, "scripts", "fibo_register.py"), run_name="fibo_register_check")
# runpy ran both steps; the loader step needs modules.sd_models (full UI) and may fail here: report it
print("state after register:", hooks.STATE)
reg["hooks"].STATE.update(off=False, why="")
from test_detect import _find_q8
q8 = _find_q8(); print("Q8:", q8)
class Shape:
    def __init__(self, s): self.shape = tuple(s)
head = gguf_names.read_header(q8)
sd = loader.preprocess_state_dict({k: Shape(v[0]) for k, v in head["tensors"].items()})
g = huggingface_guess.guess(sd)
print("guess:", type(g).__name__, "repo:", g.huggingface_repo, "prefix:", g.unet_key_prefix, "unet_config left:", g.unet_config)
eng = [M.__name__ for M in loader.possible_models if type(g) in getattr(M, "matched_guesses", [])]
print("engine picked:", eng[:1])
print("model_type:", g.model_type, "latent:", type(g.latent_format).__name__ if not isinstance(g.latent_format, type) else g.latent_format.__name__)
tok = loader.load_huggingface_component(g, "tokenizer", "transformers", "Qwen2Tokenizer", g.huggingface_repo, None)
print("tokenizer:", type(tok).__name__, "ids for 'hello':", tok("hello")["input_ids"])
print("vision pad id:", tok.convert_tokens_to_ids("<|image_pad|>"))
