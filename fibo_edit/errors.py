# User-facing refusals. One short plain line each. Author: Stable Yogi.

TAG = "[FIBO Edit]"


class FiboRefusal(Exception):
    """A planned refusal: show the message, stop this job, keep the host alive."""


NOT_THIS_MODEL = "The loaded checkpoint is not FIBO 2.0 Edit. Load a FIBO 2.0 Edit file or turn the panel off."
NO_PICTURE = "FIBO 2.0 Edit needs a picture. Put one in the img2img box."
EMPTY_PROMPT = "Write an edit instruction in the prompt box."
TOO_MANY = "FIBO 2.0 Edit takes 1 to 5 pictures; you gave {n}."
MASK_MANY = "A mask works only with exactly one picture. Remove the extra pictures or the mask."
MASK_SIZE = "The mask size {m} does not match the picture size {p}."
FILE_MISSING = "No {what} file found. Searched: {where}"
WRONG_VAE = "The VAE chosen in the top bar is not a FLUX.2 VAE ({name}). Pick flux2-vae there, or clear the VAE box so the extension finds it."
OUT_OF_MEMORY = "Out of memory during {stage}. Loaded parts were freed; try a smaller size."
INERT = "Panel off: no FIBO 2.0 Edit model is loaded. Pick the FIBO file as Checkpoint, then press Generate once; it loads then."
DISABLED = "FIBO Edit turned itself off after an error: {why}"
PANEL_OFF = "Turn on the FIBO 2.0 Edit panel in img2img to use this checkpoint."


LAST = {"message": ""}  # the panel status line shows the last message


def say(msg: str) -> str:
    LAST["message"] = msg
    line = f"{TAG} {msg}"
    print(line)
    return line


def refuse(msg: str):
    """Print the refusal once and stop the job."""
    say(msg)
    raise FiboRefusal(msg)
