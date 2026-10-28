"""Text-to-image graphs: Z-Image Turbo (fast, default) and Qwen-Image 2512 (best typography)."""

NEG = "blurry, lowres, watermark, logo, text artifacts, deformed, oversaturated, plastic, CGI render"

ENGINES = {
    "zimage": dict(unet="z_image_turbo_bf16.safetensors", clip="qwen_3_4b.safetensors", clip_type="lumina2",
                   vae="ae.safetensors", steps=8, cfg=1.0, shift=3.0),
    "qwen": dict(unet="qwen_image_2512_bf16.safetensors", clip="qwen_2.5_vl_7b.safetensors", clip_type="qwen_image",
                 vae="qwen_image_vae.safetensors", steps=20, cfg=2.5, shift=3.1),
}


def text_to_image(prompt: str, *, width=1024, height=1024, seed=0, engine="zimage", negative=NEG, prefix="adfield"):
    """Output node: "10"."""
    e = ENGINES[engine]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": e["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": e["clip"], "type": e["clip_type"], "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": e["vae"]}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": e["shift"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
              "latent_image": ["7", 0], "seed": seed, "steps": e["steps"], "cfg": e["cfg"],
              "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }
