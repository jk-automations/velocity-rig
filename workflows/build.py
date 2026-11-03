#!/usr/bin/env python
"""Generate the Velocity Rig ComfyUI workflows (UI + API format).

  ~/ComfyUI/venv/bin/python workflows/velocity_rig/build.py [--out DIR] [--object-info FILE]

velocity_rig_race   car + rival -> Velocity Plate (Codex / Nano Banana) -> LTX-2.5 image to video (+ audio) -> Velocity FX
velocity_rig_shot   car + rival -> Z-Image Turbo first frame -> LTX-2.5 image to video (+ audio) -> Velocity FX
velocity_rig_photo  your car photo -> Qwen Image Edit puts it into the rig's framing -> LTX-2.5 -> Velocity FX
velocity_rig_reel   Velocity Reel: stitch the last N saved shots into one hard-cut reel with their LTX audio
All shot workflows run one shot per execution; set shot_index to 'increment' and queue N runs for N consecutive race shots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "workflows" / "full"))
from build_workflows import LTX, QWEN, SIG1, SIG2, UI_WIDGETS, load_pack  # noqa: E402
from wfkit import WF, fetch_object_info, spec_from_class  # noqa: E402

VR_COL = ("#7a3b12", "#5a2a0c")  # Velocity Rig nodes
ZIMG = dict(unet="z_image_turbo_bf16.safetensors", clip="qwen_3_4b.safetensors", vae="ae.safetensors")
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

INTRO = """# Velocity Rig: fast-paced camera-rig car race
Pick a **hero car**, a **rival car**, a **world** and a **camera rig**, press **Run**: one 2-4 s shot with LTX-2.5's own engine sound.

**Rigs**: wheel cam (front / rear), door mount, bumper cam, rear bumper (reverse), roof cam, exhaust close-up,
arm car chase, arm car orbit, FPV drone dive / chase, top-down drone, helicopter wide, roadside whip pan,
low pass-by, crash zoom. **Worlds**: mountain pass, coastal cliffs, city night / day / tunnel, desert canyon,
forest gravel, snow mountain, rain docklands, race circuit, or type your own.

**Full race sequence**: leave rig on *auto (race sequence)*, set **shot_index** to *increment*, then queue 6-8 runs.
Every run advances the race arc (launch -> bumper draft -> hard brake -> wheel-cam hairpin -> FPV drift -> door-mount overtake -> reverse bumper defense -> whip-pan finish) and switches camera rigs."""


def extra_specs():
    out = {}
    m = load_pack("comfyui-velocity-rig", "comfyui_velocity_rig")
    for name, cls in m.NODE_CLASS_MAPPINGS.items():
        out[name] = spec_from_class(cls)
    return out


def rig_block(wf, g, **kw):
    wf.note(g, "Velocity Rig", INTRO, width=520)
    rig = wf.add("VelocityRig", g, title="Velocity Rig (your shot)", color=VR_COL, size=[420, 620], **kw)
    wf.add("PreviewAny", g, title="Shot", source=rig.o("label"))
    wf.add("PreviewAny", g, title="Motion prompt (LTX)", source=rig.o("motion_prompt"))
    return rig


BUILDER_INTRO = """# Velocity Rig Builder: interactive camera-rig sandbox
Build any camera rig for ANY subject archetype and visual style.
Pick a **subject archetype** (car, motorcycle, jet, action sports, speedboat), an **animation / visual style** (photoreal, anime, cartoon 3D, synthwave, cyberpunk, comic), a **mount anchor**, **camera pitch**, **lens profile**, and **mount stability physics**.
Connects seamlessly to Velocity Plate / Z-Image Turbo, LTX-2.5, and Velocity FX."""


def rig_builder_block(wf, g, **kw):
    wf.note(g, "Velocity Rig Builder", BUILDER_INTRO, width=520)
    rig = wf.add("VelocityRigBuilder", g, title="Velocity Rig Builder", color=VR_COL, size=[440, 720], **kw)
    wf.add("PreviewAny", g, title="Shot", source=rig.o("label"))
    wf.add("PreviewAny", g, title="Motion prompt (LTX)", source=rig.o("motion_prompt"))
    return rig


def codex_plate(wf, g, rig):
    wf.note(g, "First frame (Codex / Nano Banana)",
            "**Velocity Plate** generates the camera-rig opening frame off-GPU via **Codex** or **Nano Banana (Gemini)**, "
            "keeping 100% of Mac Studio VRAM free for LTX-2.5 video + audio.")
    plate = wf.add("VelocityPlate", g, title="Velocity Plate (Codex / Nano Banana)", color=VR_COL,
                   plate_prompt=rig.o("plate_prompt"), width=rig.o("plate_width"), height=rig.o("plate_height"),
                   backend="codex", seed=rig.o("seed"))
    wf.add("SaveImage", g, title="First frame", images=plate.o(0), filename_prefix="velocity_rig/frame")
    return plate


def zimage_plate(wf, g, rig):
    wf.note(g, "First frame", "**Z-Image Turbo** (8 steps) paints the opening frame exactly the way the rig sees the car: "
            "on the wheel arch, on the door, from the drone... LTX animates from this frame, so a strong first frame = a strong shot.")
    unet = wf.add("UNETLoader", g, title="Z-Image Turbo", unet_name=ZIMG["unet"], weight_dtype="default")
    clip = wf.add("CLIPLoader", g, clip_name=ZIMG["clip"], type="lumina2", device="default")
    vae = wf.add("VAELoader", g, vae_name=ZIMG["vae"])
    ms = wf.add("ModelSamplingAuraFlow", g, model=unet.o(0), shift=3.0)
    pos = wf.add("CLIPTextEncode", g, title="First-frame prompt", clip=clip.o(0), text=rig.o("plate_prompt"))
    neg = wf.add("CLIPTextEncode", g, title="First-frame negative", clip=clip.o(0), text=rig.o("plate_negative"))
    lat = wf.add("EmptySD3LatentImage", g, width=rig.o("plate_width"), height=rig.o("plate_height"), batch_size=1)
    ks = wf.add("KSampler", g, title="Z-Image (8 steps)", model=ms.o(0), positive=pos.o(0), negative=neg.o(0), latent_image=lat.o(0),
                seed=rig.o("seed"), steps=8, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    dec = wf.add("VAEDecode", g, samples=ks.o(0), vae=vae.o(0))
    wf.add("SaveImage", g, title="First frame", images=dec.o(0), filename_prefix="velocity_rig/frame")
    return dec


def photo_plate(wf, g, rig, rival=False):
    if rival:
        wf.note(g, "Identity refs", "**Qwen Image Edit 2511** (Lightning, 4 steps) shoots every frame from two identity refs, "
                "**hero** (image 1) and **rival** (image 2), so both vehicles keep one design shot to shot. Make the refs once with "
                "Z-Image Turbo (velocity_rig_shot) or drop in your own photos. Open weights only.")
        photo = wf.add("LoadImage", g, title="Hero reference", image="velocity_rig_hero.png")
        rival_img = wf.add("LoadImage", g, title="Rival reference", image="velocity_rig_rival.png")
        lead = ("A completely new photograph. Image 1 is the hero vehicle; image 2 is the rival vehicle. Keep each vehicle's exact "
                "body shape, paint, markings, number and wheels. The camera and scene are:")
        refs = dict(image1=photo.o(0), image2=rival_img.o(0))
    else:
        wf.note(g, "Your car", "**Qwen Image Edit 2511** (Lightning, 4 steps) re-shoots your car photo from the rig's position: same body, "
                "paint and wheels, new camera, new world. Use one clean photo of the car for every shot so it stays the same car.")
        photo = wf.add("LoadImage", g, title="Your car photo", image="adfield_demo_car.png")
        lead = ("Show the exact car from image 1, with the same body shape, paint, "
                "wheels and number, in a completely new photograph. The camera and scene are:")
        refs = dict(image1=photo.o(0))
    unet = wf.add("UNETLoader", g, title="Qwen Image Edit 2511", unet_name=QWEN["unet"], weight_dtype="default")
    clip = wf.add("CLIPLoader", g, clip_name=QWEN["clip"], type="qwen_image", device="default")
    vae = wf.add("VAELoader", g, vae_name=QWEN["vae"])
    aura = wf.add("ModelSamplingAuraFlow", g, model=unet.o(0), shift=3.1)
    cfgn = wf.add("CFGNorm", g, model=aura.o(0), strength=1.0)
    light = wf.add("LoraLoaderModelOnly", g, title="Lightning 4-step LoRA", model=cfgn.o(0), lora_name=LIGHTNING, strength_model=1.0)
    ask = wf.add("StringConcatenate", g, title="Edit instruction", string_a=lead, string_b=rig.o("plate_prompt"), delimiter=" ")
    pos = wf.add("TextEncodeQwenImageEditPlus", g, title="Edit prompt", clip=clip.o(0), prompt=ask.o(0), vae=vae.o(0), **refs)
    pm = wf.add("FluxKontextMultiReferenceLatentMethod", g, conditioning=pos.o(0), reference_latents_method="index_timestep_zero")
    neg = wf.add("TextEncodeQwenImageEditPlus", g, title="Edit negative", clip=clip.o(0), prompt=rig.o("plate_negative"), vae=vae.o(0), **refs)
    nm = wf.add("FluxKontextMultiReferenceLatentMethod", g, conditioning=neg.o(0), reference_latents_method="index_timestep_zero")
    lat = wf.add("EmptySD3LatentImage", g, width=rig.o("plate_width"), height=rig.o("plate_height"), batch_size=1)
    ks = wf.add("KSampler", g, title="Qwen edit (4 steps)", model=light.o(0), positive=pm.o(0), negative=nm.o(0), latent_image=lat.o(0),
                seed=rig.o("seed"), steps=4, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    dec = wf.add("VAEDecode", g, samples=ks.o(0), vae=vae.o(0))
    wf.add("SaveImage", g, title="First frame", images=dec.o(0), filename_prefix="velocity_rig/frame")
    return dec


def ltx_block(wf, g, rig, plate):
    wf.note(g, "Motion", "**LTX-2.5 22B distilled**, image to video in two stages (half-res pass, x2 latent upscale, refine), "
            "with **synchronized audio**: the engine, tires and wind come from the same model. Size, length and fps all come from the rig node.", width=560)
    unet = wf.add("UNETLoader", g, title="LTX-2.5 22B distilled", unet_name=LTX["unet"], weight_dtype="default")
    vvae = wf.add("VAELoader", g, title="LTX video VAE", vae_name=LTX["vvae"])
    avae = wf.add("VAELoader", g, title="LTX audio VAE", vae_name=LTX["avae"])
    clip = wf.add("CLIPLoader", g, title="Gemma 4 text encoder", clip_name=LTX["clip"], type="ltxv", device="default")
    ups = wf.add("LatentUpscaleModelLoader", g, model_name=LTX["ups"])
    sampler = wf.add("KSamplerSelect", g, sampler_name="euler_ancestral")
    sig1 = wf.add("ManualSigmas", g, title="Stage 1 sigmas", sigmas=SIG1)
    sig2 = wf.add("ManualSigmas", g, title="Stage 2 sigmas", sigmas=SIG2)
    pos = wf.add("CLIPTextEncode", g, title="Motion prompt", clip=clip.o(0), text=rig.o("motion_prompt"))
    neg = wf.add("CLIPTextEncode", g, title="Motion negative", clip=clip.o(0), text=rig.o("video_negative"))
    cond = wf.add("LTXVConditioning", g, positive=pos.o(0), negative=neg.o(0), frame_rate=rig.o("fps"))
    alat = wf.add("LTXVEmptyLatentAudio", g, audio_vae=avae.o(0), frames_number=rig.o("frames"), frame_rate=rig.o("fps"), batch_size=1)
    vlat = wf.add("EmptyLTXVLatentVideo", g, title="Half-res latent", width=rig.o("half_width"), height=rig.o("half_height"), length=rig.o("frames"), batch_size=1)
    fit1 = wf.add("ImageScale", g, title="Fit frame (half-res)", image=plate.o(0), upscale_method="lanczos", width=rig.o("half_width"), height=rig.o("half_height"), crop="center")
    pre1 = wf.add("LTXVPreprocess", g, image=fit1.o(0), img_compression=18)
    in1 = wf.add("LTXVImgToVideoInplace", g, title="First frame (stage 1)", vae=vvae.o(0), image=pre1.o(0), latent=vlat.o(0), strength=0.7, bypass=False)
    av1 = wf.add("LTXVConcatAVLatent", g, video_latent=in1.o(0), audio_latent=alat.o(0))
    n1 = wf.add("RandomNoise", g, noise_seed=rig.o("seed"))
    g1 = wf.add("LTXVDualCFGGuider", g, model=unet.o(0), positive=cond.o(0), negative=cond.o(1), video_cfg=1.0, audio_cfg=1.0)
    s1 = wf.add("SamplerCustomAdvanced", g, title="LTX stage 1", noise=n1.o(0), guider=g1.o(0), sampler=sampler.o(0), sigmas=sig1.o(0), latent_image=av1.o(0))
    sp1 = wf.add("LTXVSeparateAVLatent", g, av_latent=s1.o(0))
    up = wf.add("LTXVLatentUpsampler", g, title="x2 latent upscale", samples=sp1.o(0), upscale_model=ups.o(0), vae=vvae.o(0))
    fit2 = wf.add("ImageScale", g, title="Fit frame (full-res)", image=plate.o(0), upscale_method="lanczos", width=rig.o("width"), height=rig.o("height"), crop="center")
    pre2 = wf.add("LTXVPreprocess", g, image=fit2.o(0), img_compression=18)
    in2 = wf.add("LTXVImgToVideoInplace", g, title="First frame (stage 2)", vae=vvae.o(0), image=pre2.o(0), latent=up.o(0), strength=1.0, bypass=False)
    av2 = wf.add("LTXVConcatAVLatent", g, video_latent=in2.o(0), audio_latent=sp1.o(1))
    n2 = wf.add("RandomNoise", g, noise_seed=42)
    g2 = wf.add("LTXVDualCFGGuider", g, model=unet.o(0), positive=cond.o(0), negative=cond.o(1), video_cfg=1.0, audio_cfg=1.0)
    s2 = wf.add("SamplerCustomAdvanced", g, title="LTX stage 2 (refine)", noise=n2.o(0), guider=g2.o(0), sampler=sampler.o(0), sigmas=sig2.o(0), latent_image=av2.o(0))
    sp2 = wf.add("LTXVSeparateAVLatent", g, av_latent=s2.o(0))
    vdec = wf.add("VAEDecodeTiled", g, samples=sp2.o(0), vae=vvae.o(0), tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)
    adec = wf.add("LTXVAudioVAEDecode", g, samples=sp2.o(1), audio_vae=avae.o(0))
    return vdec, adec


def fx_block(wf, g, rig, vdec, adec):
    wf.note(g, "Speed FX", "**Velocity FX** adds what a real rig adds: vibration, radial speed blur that keeps the car sharp, crash zoom, "
            "lens fringing, mount vibration, road-bump jolts, chase-operator lag and sensor grain, using each rig's own defaults. "
            "Sliders at -1 follow the rig; strength 0 bypasses it. **Motion Trim** then cuts the shot to its action (LTX holds the "
            "first frame and drifts at the end) and speeds it up 1.2x, audio included. Both the raw and the finished clip are saved.")
    raw = wf.add("CreateVideo", g, title="Raw clip", images=vdec.o(0), audio=adec.o(0), fps=rig.o("fps"))
    wf.add("SaveVideo", g, title="Save raw", video=raw.o(0), filename_prefix="velocity_rig/raw", format="mp4", codec="h264")
    fx = wf.add("VelocityFX", g, title="Velocity FX (speed)", color=VR_COL, images=vdec.o(0), fx=rig.o("fx"), fps=rig.o("fps"), seed=rig.o("seed"), strength=1.0)
    trim = wf.add("VelocityMotionTrim", g, title="Motion Trim (cut to the action)", color=VR_COL, images=fx.o(0), audio=adec.o(0),
                  measure=vdec.o(0), fps=rig.o("fps"), threshold=0.25, pad=3, speed=1.2, head=4)
    vid = wf.add("CreateVideo", g, title="Shot with FX", images=trim.o(0), audio=trim.o(1), fps=rig.o("fps"))
    wf.add("SaveVideo", g, title="Save shot", video=vid.o(0), filename_prefix="velocity_rig/shot", format="mp4", codec="h264")


def build(oi, extra, mode="zimage", use_builder=False, **rig_kw):
    wf = WF(oi, extra, lenient=(mode in ("photo", "refs", "codex")))
    for name, col, nc in (("Shot", "#7a3b12", 1), ("First frame", "#b58b2a", 4), ("Motion", "#a1309b", 5), ("Speed FX", "#3f8f5f", 3)):
        wf.group(name, col, nc)
    rig = rig_builder_block(wf, "Shot", **rig_kw) if use_builder else rig_block(wf, "Shot", **rig_kw)
    if mode in ("photo", "refs"):
        plate = photo_plate(wf, "First frame", rig, rival=(mode == "refs"))
    elif mode == "codex":
        plate = codex_plate(wf, "First frame", rig)
    else:
        plate = zimage_plate(wf, "First frame", rig)
    vdec, adec = ltx_block(wf, "Motion", rig, plate)
    fx_block(wf, "Speed FX", rig, vdec, adec)
    return wf


def build_reel(oi, extra):
    wf = WF(oi, extra)
    wf.group("Reel", "#3f8f5f", 2)
    wf.note("Reel", "Velocity Reel", "# Velocity Reel\nQueue your shots with **velocity_rig_race** or **velocity_rig_shot** (shot_index on *increment*), then run this. "
            "It takes the last **count** shots saved as `velocity_rig/shot_*.mp4`, oldest first, trims the opening frames "
            "of each one (LTX starts on the still) and hard-cuts them together with their engine audio.", width=460)
    r = wf.add("VelocityReel", "Reel", title="Velocity Reel", color=VR_COL, prefix="velocity_rig/shot", count=6, trim_frames=6, fps=24)
    wf.add("PreviewAny", "Reel", title="Shots used", source=r.o("shots"))
    wf.add("SaveVideo", "Reel", title="Save reel", video=r.o("reel"), filename_prefix="velocity_rig/reel", format="mp4", codec="h264")
    return wf


def write(wf, outdir: Path, stem: str):
    api = wf.to_api()
    (outdir / f"{stem}_api.json").write_text(json.dumps(api, indent=1))
    ui = wf.to_ui(UI_WIDGETS)
    (outdir / f"{stem}.json").write_text(json.dumps(ui, indent=1))
    print(f"{stem}: {len(wf.nodes)} nodes ({len(api)} in the API graph), {len(ui['links'])} links")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--object-info")
    a = ap.parse_args()
    oi = json.load(open(a.object_info)) if a.object_info else fetch_object_info()
    extra = extra_specs()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write(build(oi, extra, mode="refs"), out, "velocity_rig_race")
    write(build(oi, extra, mode="zimage"), out, "velocity_rig_shot")
    write(build(oi, extra, mode="photo", car="the car in image 1"), out, "velocity_rig_photo")
    write(build(oi, extra, mode="refs", use_builder=True), out, "velocity_rig_builder")
    write(build_reel(oi, extra), out, "velocity_rig_reel")


if __name__ == "__main__":
    main()
