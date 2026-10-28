"""LTX-2.5 22B distilled video graphs (video + synchronized audio) for ComfyUI, API format.

All builders return an API-format graph dict. Output node id is always "save" (SaveVideo, mp4/h264).
  ltx_t2v(prompt, ...)   text to video.            Output: "save"
  ltx_i2v(prompt, ...)   image to video.           Input LoadImage "img1" (first frame); output "save"
  ltx_flf(prompt, ...)   first + last frame.       Inputs LoadImage "img1" (start), "img2" (end); output "save"

Settings follow ComfyUI's bundled LTX-2.5 templates: euler_ancestral, manual distilled sigmas, cfg 1.
t2v/i2v are two-stage (half-res pass, x2 latent upscale, refine pass); flf is a single full-res pass
with start/end frame guides (as in the official first-last-frame template). Sizes are rounded to
multiples of 64 for the two-stage graphs (half-res must be a multiple of 32) and 32 for flf.
Length is seconds*fps+1 frames, so seconds*fps should be a multiple of 8 (24 fps: any whole second).
"""
from __future__ import annotations

UNET = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
VVAE = "ltx-2.5-video-vae-bf16.safetensors"
AVAE = "ltx-2.5-audio-vae-bf16.safetensors"
CLIP = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
UPS = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
NEG = "pc game, console game, video game, cartoon, childish, ugly"
SIG1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIG2 = "0.85, 0.7250, 0.4219, 0.0"


def _round(v: int, m: int) -> int:
    return max(m, int(round(v / m)) * m)


def _base(prompt, negative, fps, length, seed, prefix, audio):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": VVAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "ltxv", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEG if negative is None else negative}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": fps}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["avae", 0], "frames_number": length, "frame_rate": fps, "batch_size": 1}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
    }
    return g


def _tail(g, final_video, final_audio, fps, prefix, audio):
    g["vdec"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": final_video, "vae": ["vvae", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}}
    vid = {"images": ["vdec", 0], "fps": fps}
    if audio:
        g["adec"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": final_audio, "audio_vae": ["avae", 0]}}
        vid["audio"] = ["adec", 0]
    g["vid"] = {"class_type": "CreateVideo", "inputs": vid}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["vid", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}}
    return g


def _two_stage(g, video_lat_fn, fps, seed, w, h, length):
    """video_lat_fn(stage, latent_ref, w, h) -> latent_ref after optional image conditioning."""
    g["ups"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPS}}
    g["vlat"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w // 2, "height": h // 2, "length": length, "batch_size": 1}}
    v1 = video_lat_fn(1, ["vlat", 0], w // 2, h // 2)
    g["av1"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": v1, "audio_latent": ["alat", 0]}}
    g["noise1"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sig1"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG1}}
    g["guide1"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "video_cfg": 1, "audio_cfg": 1}}
    g["samp1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise1", 0], "guider": ["guide1", 0], "sampler": ["sampler", 0], "sigmas": ["sig1", 0], "latent_image": ["av1", 0]}}
    g["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp1", 0]}}
    g["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep1", 0], "upscale_model": ["ups", 0], "vae": ["vvae", 0]}}
    v2 = video_lat_fn(2, ["up", 0], w, h)
    g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": v2, "audio_latent": ["sep1", 1]}}
    g["noise2"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}
    g["sig2"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG2}}
    g["guide2"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "video_cfg": 1, "audio_cfg": 1}}
    g["samp2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise2", 0], "guider": ["guide2", 0], "sampler": ["sampler", 0], "sigmas": ["sig2", 0], "latent_image": ["av2", 0]}}
    g["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp2", 0]}}
    return ["sep2", 0], ["sep2", 1]


def ltx_t2v(prompt, *, width=768, height=448, seconds=3, fps=24, seed=0, audio=True, strength=1.0,
            negative=None, prefix="clip"):
    """LTX-2.5 text to video, two-stage. No image inputs. Output node: "save" (SaveVideo mp4)."""
    w, h = _round(width, 64), _round(height, 64)
    length = int(seconds * fps) + 1
    g = _base(prompt, negative, fps, length, seed, prefix, audio)
    fv, fa = _two_stage(g, lambda stage, ref, ww, hh: ref, fps, seed, w, h, length)
    return _tail(g, fv, fa, fps, prefix, audio)


def ltx_i2v(prompt, *, width=768, height=448, seconds=3, fps=24, seed=0, audio=True, strength=1.0,
            negative=None, prefix="clip"):
    """LTX-2.5 image to video, two-stage. Input: LoadImage node "img1" = first frame (cover-cropped
    to the target size inside the graph). Output node: "save" (SaveVideo mp4).
    strength (0-1) scales how hard the first frame is locked: stage 1 uses 0.7*strength (template
    default 0.7), stage 2 uses strength."""
    w, h = _round(width, 64), _round(height, 64)
    length = int(seconds * fps) + 1
    g = _base(prompt, negative, fps, length, seed, prefix, audio)
    g["img1"] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}

    def cond(stage, ref, ww, hh):
        g[f"fit{stage}"] = {"class_type": "ImageScale", "inputs": {"image": ["img1", 0], "upscale_method": "lanczos", "width": ww, "height": hh, "crop": "center"}}
        g[f"pre{stage}"] = {"class_type": "LTXVPreprocess", "inputs": {"image": [f"fit{stage}", 0], "img_compression": 18}}
        g[f"inpl{stage}"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["vvae", 0], "image": [f"pre{stage}", 0], "latent": ref,
            "strength": round(min(1.0, 0.7 * strength if stage == 1 else strength), 4), "bypass": False}}
        return [f"inpl{stage}", 0]

    fv, fa = _two_stage(g, cond, fps, seed, w, h, length)
    return _tail(g, fv, fa, fps, prefix, audio)


def _ltx_flf_single(prompt, *, width=768, height=448, seconds=3, fps=24, seed=0, audio=True, strength=1.0,
                    negative=None, prefix="clip"):
    """LTX-2.5 first+last frame to video, single full-res pass (official FLF template layout).
    Inputs: LoadImage "img1" (start frame), "img2" (end frame); both cover-cropped to size.
    Output node: "save" (SaveVideo mp4). strength scales the 0.7 guide strength of both frames."""
    w, h = _round(width, 32), _round(height, 32)
    length = int(seconds * fps) + 1
    g = _base(prompt, negative, fps, length, seed, prefix, audio)
    g["vlat"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
    prev_pos, prev_neg, prev_lat = ["cond", 0], ["cond", 1], ["vlat", 0]
    for i, idx in (("img1", 0), ("img2", -1)):
        g[i] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
        g[f"fit_{i}"] = {"class_type": "ImageScale", "inputs": {"image": [i, 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}}
        g[f"pre_{i}"] = {"class_type": "LTXVPreprocess", "inputs": {"image": [f"fit_{i}", 0], "img_compression": 18}}
        g[f"guide_{i}"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": prev_pos, "negative": prev_neg, "vae": ["vvae", 0], "latent": prev_lat,
            "image": [f"pre_{i}", 0], "frame_idx": idx, "strength": round(min(1.0, 0.7 * strength), 4)}}
        prev_pos, prev_neg, prev_lat = [f"guide_{i}", 0], [f"guide_{i}", 1], [f"guide_{i}", 2]
    g["av1"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": prev_lat, "audio_latent": ["alat", 0]}}
    g["noise1"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sig1"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG1}}
    g["guide1"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": prev_pos, "negative": prev_neg, "video_cfg": 1, "audio_cfg": 1}}
    g["samp1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise1", 0], "guider": ["guide1", 0], "sampler": ["sampler", 0], "sigmas": ["sig1", 0], "latent_image": ["av1", 0]}}
    g["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp1", 1]}}
    g["crop"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": prev_pos, "negative": prev_neg, "latent": ["sep1", 0]}}
    return _tail(g, ["crop", 2], ["sep1", 1], fps, prefix, audio)


def ltx_flf(prompt, *, width=768, height=448, seconds=3, fps=24, seed=0, audio=True, strength=1.0,
            negative=None, prefix="clip", single_pass=False):
    """LTX-2.5 first+last frame to video, two-stage like ltx_i2v: half-res pass with start/end frame
    guides, x2 latent upscale, refine pass with the guides re-added at full res.
    Inputs: LoadImage "img1" (start frame), "img2" (end frame); both cover-cropped to size.
    Output node: "save" (SaveVideo mp4). strength scales the 0.7 guide strength of both frames.
    single_pass=True gives the official template's single full-res pass instead."""
    if single_pass:
        return _ltx_flf_single(prompt, width=width, height=height, seconds=seconds, fps=fps, seed=seed,
                               audio=audio, strength=strength, negative=negative, prefix=prefix)
    w, h = _round(width, 64), _round(height, 64)
    length = int(seconds * fps) + 1
    gs = round(min(1.0, 0.7 * strength), 4)
    g = _base(prompt, negative, fps, length, seed, prefix, audio)
    g["img1"] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
    g["img2"] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
    g["ups"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPS}}

    def guides(stage, pos, neg, lat, ww, hh):
        for i, idx in (("img1", 0), ("img2", -1)):
            n = f"{i}_s{stage}"
            g["fit_" + n] = {"class_type": "ImageScale", "inputs": {"image": [i, 0], "upscale_method": "lanczos", "width": ww, "height": hh, "crop": "center"}}
            g["pre_" + n] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["fit_" + n, 0], "img_compression": 18}}
            g["guide_" + n] = {"class_type": "LTXVAddGuide", "inputs": {
                "positive": pos, "negative": neg, "vae": ["vvae", 0], "latent": lat,
                "image": ["pre_" + n, 0], "frame_idx": idx, "strength": gs}}
            pos, neg, lat = ["guide_" + n, 0], ["guide_" + n, 1], ["guide_" + n, 2]
        return pos, neg, lat

    g["vlat"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w // 2, "height": h // 2, "length": length, "batch_size": 1}}
    p1, n1, l1 = guides(1, ["cond", 0], ["cond", 1], ["vlat", 0], w // 2, h // 2)
    g["av1"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": l1, "audio_latent": ["alat", 0]}}
    g["noise1"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sig1"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG1}}
    g["guide1"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": p1, "negative": n1, "video_cfg": 1, "audio_cfg": 1}}
    g["samp1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise1", 0], "guider": ["guide1", 0], "sampler": ["sampler", 0], "sigmas": ["sig1", 0], "latent_image": ["av1", 0]}}
    g["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp1", 1]}}
    g["crop1"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": p1, "negative": n1, "latent": ["sep1", 0]}}
    g["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["crop1", 2], "upscale_model": ["ups", 0], "vae": ["vvae", 0]}}
    p2, n2, l2 = guides(2, ["crop1", 0], ["crop1", 1], ["up", 0], w, h)
    g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": l2, "audio_latent": ["sep1", 1]}}
    g["noise2"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}
    g["sig2"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG2}}
    g["guide2"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": p2, "negative": n2, "video_cfg": 1, "audio_cfg": 1}}
    g["samp2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise2", 0], "guider": ["guide2", 0], "sampler": ["sampler", 0], "sigmas": ["sig2", 0], "latent_image": ["av2", 0]}}
    g["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp2", 1]}}
    g["crop2"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": p2, "negative": n2, "latent": ["sep2", 0]}}
    return _tail(g, ["crop2", 2], ["sep2", 1], fps, prefix, audio)
