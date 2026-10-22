# FIBO 2.0 Edit for Forge Neo

An extension for Forge Neo. It loads the FIBO 2.0 Edit model and adds a "FIBO 2.0 Edit" panel to the img2img tab.
You give it one picture (or up to five) and an edit instruction. It returns the edited picture.

Author: Stable Yogi

## Install

1. Copy the folder `sd-forge-fibo-edit` into `extensions\` inside your Forge Neo folder.
2. Restart Forge Neo. When it starts, the console shows `[FIBO Edit] FIBO 2.0 Edit registered.`

The extension adds no packages and changes no Forge files.

## Files you need

| File | Folder |
| --- | --- |
| The FIBO 2.0 Edit checkpoint (GGUF, for example Q8_0, or bf16 safetensors) | `models\Stable-diffusion` (or your diffusion model folder) |
| Qwen3-VL 4B text encoder, for example `qwen3vl_4b_fp8_scaled.safetensors` | `models\text_encoder` |
| FLUX.2 VAE, for example `flux2-vae.safetensors` | `models\VAE` |

The checkpoint holds only the image model. If you do not pick a text encoder and VAE in the top bar, the extension
finds them in those folders itself (and in the folders given with `--text-encoder-dirs` and `--vae-dirs`).
If one is missing, the console says which file it looked for and where.

## Use

1. Load the FIBO 2.0 Edit checkpoint.
2. Open the img2img tab. Put your picture in the img2img box. That is picture 1.
3. Open the "FIBO 2.0 Edit" panel and switch it on.
4. Optional: add up to 4 more pictures (pictures 2 to 5) in the panel.
5. Write the edit in the prompt box. Plain text works, for example `make the car red`. A JSON caption also works.
6. Press "Recommended settings". It sets Euler, Simple, 4 steps, CFG 1 and the right size for picture 1.
7. Press Generate.

The "Check" button shows the panel status: ready, wrong model, or the last message.

## Settings

- **Shift**: `size-based` (default) works out the noise schedule from the output size. `fixed 1.15` uses one value for every size.
- **Use my own width and height**: off (default) makes the output the size of picture 1 after it is fitted to a trained size.
  On, it uses the width and height you typed, rounded down to a multiple of 16 (at least 64).
- **CFG**: the model is made for CFG 1. A higher CFG works, but each step then runs twice and is slower.
- **Mask**: draw a mask on picture 1 (img2img inpaint) to mark the area to change. A mask works only with exactly one picture.
  "Inverted mask" is honoured. The extension does not blend or paste back; the model redraws the whole picture.
  Because the whole picture is redrawn, areas outside the mask can change a little (small drift in detail and colour).
- **Denoise**: always 1.0. The extension sets it, because the model starts from pure noise.

## When something is wrong

The extension never stops Forge. A problem gives one line in the console and in the panel status, for example:

- "The loaded checkpoint is not FIBO 2.0 Edit." - load the right checkpoint or switch the panel off.
- "FIBO 2.0 Edit needs a picture." - put a picture in the img2img box.
- "Out of memory during sampling." - try a smaller size.

If an unexpected error happens, the extension turns itself off and Forge keeps working with other models.

## Tests

With the Forge Neo venv Python (CPU only, no GPU needed):

```
venv\Scripts\python.exe -I extensions\sd-forge-fibo-edit\tests\run_all.py
```

## Licence

Non-commercial: CC BY-NC 4.0.

The FIBO 2.0 Edit model weights are under CC BY-NC 4.0 (non-commercial use only).
