"""Qwen-Image-Edit 2511 graph (bf16) with the official Lightning 4-step LoRA by default.

qwen_edit(prompt, n_refs=1, ...) -> API-format graph.
  Inputs : LoadImage nodes "img1".."img{n_refs}" (n_refs 1-3). img1 is the image being edited;
           img2/img3 are extra references (product, style, material...). Refer to them in the
           prompt as "image 1", "image 2", "image 3".
  Output : node "save" (SaveImage, PNG).
Size: if width and height are both given the output is exactly that size (rounded to 16);
otherwise img1's aspect ratio is kept at `megapixels` total. angles_lora=True loads the
multiple-angles LoRA (use prompts like "rotate the camera 45 degrees to the left").
lightning=True (default): Qwen-Image-Edit-2511-Lightning-4steps LoRA, 4 steps, cfg 1 (template's lightning branch).
lightning=False: plain schedule, 20 steps cfg 3 (template: 40 steps). steps/cfg kwargs override either.
"""
from __future__ import annotations

UNET = "qwen_image_edit_2511_bf16.safetensors"
CLIP = "qwen_2.5_vl_7b.safetensors"
VAE = "qwen_image_vae.safetensors"
ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def qwen_edit(prompt, n_refs=1, *, seed=0, megapixels=1.0, width=None, height=None,
              angles_lora=False, negative="", prefix="edit", lightning=True, steps=None, cfg=None):
    steps = steps if steps is not None else (4 if lightning else 20)
    cfg = cfg if cfg is not None else (1.0 if lightning else 3.0)
    n_refs = max(1, min(3, int(n_refs)))
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }
    model = ["unet", 0]
    if angles_lora:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": ANGLES, "strength_model": 1.0}}
        model = ["lora", 0]
    g["aura"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}}
    g["cfgnorm"] = {"class_type": "CFGNorm", "inputs": {"model": ["aura", 0], "strength": 1.0}}
    model = ["cfgnorm", 0]
    if lightning:
        g["lightning"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": LIGHTNING, "strength_model": 1.0}}
        model = ["lightning", 0]
    refs = {}
    for i in range(1, n_refs + 1):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
        refs[f"image{i}"] = [f"img{i}", 0]
    for name, text in (("pos", prompt), ("neg", negative or "")):
        g[name] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **refs}}
        g[name + "_m"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": [name, 0], "reference_latents_method": "index_timestep_zero"}}
    if width and height:
        w, h = max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)
        g["lat"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    else:
        g["fit"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["img1", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}}
        g["lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["fit", 0], "vae": ["vae", 0]}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["pos_m", 0], "negative": ["neg_m", 0], "latent_image": ["lat", 0],
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}}
    return g
