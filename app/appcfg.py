"""Velocity Rig app config: controls, library and shot preview. The generic engine is server.py."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "custom_nodes" / "comfyui-velocity-rig"
MODULE = "velocityrig"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"velocity_rig_{name}", PACK / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rigs = _load("rigs")
MODEL_PATTERNS = [("ltx-2.5", "LTX-2.5 22B distilled"), ("z_image_turbo", "Z-Image Turbo"), ("qwen_image_edit", "Qwen Image Edit 2511"),
                  ("lightning", "Qwen Lightning 4-step LoRA"), ("gemma", "Gemma text encoder"), ("qwen_3_4b", "Qwen3 4B (Z-Image text encoder)"),
                  ("qwen_2.5_vl", "Qwen2.5-VL 7B (Qwen Edit text encoder)")]

CONFIG = {
    "name": "Velocity Rig",
    "tagline": "Camera-rig simulation for high-speed vehicle shots. Pick a subject, a world and a rig; every plate, clip and engine sound comes from open weights through Comfy API v2.",
    "topline": "Open weights end to end  /  ComfyUI graphs  /  Comfy API v2",
    "accent": "#ffb020", "accent_ink": "#1d1300", "port": 8811, "plan": True, "render_label": "Render",
    "node_classes": ["^Velocity"],
    "models": [{"name": n, "match": [p]} for p, n in MODEL_PATTERNS],
    "stack": [
        {"stage": "Shot planning", "model": "Velocity Rig node", "nodes": "VelocityRig / VelocityRigBuilder", "why": "Rig library -> first-frame prompt, LTX motion prompt, sizes, frames, FX defaults."},
        {"stage": "Identity refs", "model": "Z-Image Turbo (bf16)", "nodes": "UNETLoader · CLIPLoader(qwen_3_4b) · KSampler", "why": "Draws the hero and rival vehicles once, or uses your photo."},
        {"stage": "Plates", "model": "Qwen Image Edit 2511 + Lightning 4-step LoRA", "nodes": "TextEncodeQwenImageEditPlus · LoraLoaderModelOnly · KSampler", "why": "Every first frame is edited from the refs, so the vehicle keeps one design shot to shot."},
        {"stage": "Video + audio", "model": "LTX-2.5 22B distilled", "nodes": "LTXVImgToVideoInplace · LTXVDualCFGGuider · LTXVLatentUpsampler · LTXVAudioVAEDecode", "why": "Two-stage image-to-video with synchronized engine sound; Gemma text encoder."},
        {"stage": "Rig physics", "model": "Velocity FX node", "nodes": "VelocityFX", "why": "Shake, radial speed blur, zoom punch, chroma, speed ramp. Frame count unchanged."},
        {"stage": "Reel", "model": "Velocity Reel node", "nodes": "VelocityReel", "why": "Hard cuts, wake-up frames trimmed, LTX audio kept."},
    ],
    "nodes": [{"name": "Velocity Rig", "what": "preset rig -> prompts, sizes, FX"}, {"name": "Velocity Rig Builder", "what": "mount, pitch, lens, stability"},
              {"name": "Velocity Rig Library", "what": "library as text"}, {"name": "Velocity FX", "what": "speed effects"}, {"name": "Velocity Reel", "what": "stitch shots"}],
    "uploads": [
        {"id": "photo", "label": "Your vehicle photo (hero identity)", "flag": "--photo", "help": "Optional. Qwen Image Edit copies its body, paint and number."},
        {"id": "rival_photo", "label": "Rival vehicle photo", "flag": "--rival-photo", "help": "Optional."},
        {"id": "pilot_photo", "label": "Pilot selfie (cockpit)", "flag": "--pilot-photo", "help": "Optional. Keeps your face in the cockpit."},
    ],
    "controls": [
        {"id": "title", "label": "Run name", "type": "text", "default": "", "group": "Shot", "placeholder": "optional"},
        {"id": "subject_type", "label": "Subject", "type": "select", "options": "subjects", "default": "car", "flag": "--subject-type", "group": "Shot"},
        {"id": "style", "label": "Visual style", "type": "select", "options": "styles", "default": "photoreal", "flag": "--style", "group": "Shot"},
        {"id": "world", "label": "World", "type": "select", "options": "worlds", "default": "mountain pass", "flag": "--world", "group": "Shot"},
        {"id": "rig", "label": "Camera rig", "type": "select", "options": "rigs", "default": "auto (race sequence)", "flag": "--rig", "group": "Shot"},
        {"id": "shots", "label": "Shots", "type": "number", "min": 1, "max": 12, "step": 1, "default": 4, "flag": "--shots", "group": "Shot"},
        {"id": "seconds", "label": "Seconds per shot (0 = rig default)", "type": "number", "min": 0, "max": 8, "step": 1, "default": 0, "flag": "--seconds", "group": "Shot"},
        {"id": "action", "label": "Action beat", "type": "select", "options": "actions", "default": "auto", "flag": "--action", "group": "Look"},
        {"id": "light", "label": "Light", "type": "select", "options": "lights", "default": "world default", "flag": "--light", "group": "Look"},
        {"id": "aspect", "label": "Aspect", "type": "select", "options": "aspects", "default": "16:9 landscape", "flag": "--aspect", "group": "Look"},
        {"id": "intensity", "label": "Speed intensity", "type": "range", "min": 0, "max": 1, "step": 0.05, "default": 0.85, "flag": "--intensity", "group": "Look"},
        {"id": "seed", "label": "Seed", "type": "number", "min": 0, "max": 2147483647, "default": 7, "flag": "--seed", "group": "Look"},
        {"id": "car", "label": "Hero vehicle", "type": "textarea", "default": "", "flag": "--car", "group": "Vehicles", "wide": True, "placeholder": "blank = the subject's default original design"},
        {"id": "rival", "label": "Rival vehicle", "type": "textarea", "default": "", "flag": "--rival", "group": "Vehicles", "wide": True, "placeholder": "blank = default rival, 'none' = solo run"},
        {"id": "world_custom", "label": "Custom location", "type": "text", "default": "", "flag": "--world-custom", "group": "Vehicles", "wide": True},
        {"id": "extra", "label": "Extra detail", "type": "text", "default": "", "flag": "--extra", "group": "Vehicles", "wide": True},
        {"id": "plates_only", "label": "Plates only", "type": "toggle", "default": False, "flag": "--plates-only", "group": "Run", "help": "Stop after the plates and contact sheet (about 1 min per plate on an M3 Ultra)."},
    ],
}


def library():
    r = rigs
    opt = lambda items: [{"v": k, "l": k} for k in items]
    return {
        "subjects": [{"v": k, "l": v["label"]} for k, v in r.ARCHETYPES.items()],
        "styles": [{"v": k, "l": v["label"]} for k, v in r.STYLES.items()],
        "worlds": opt(r.WORLDS),
        "rigs": [{"v": k, "l": k} for k in ("auto (race sequence)", "auto (cycle by shot)", "random")] + [{"v": k, "l": f"{k}  ({r.GROUPS[v['group']]}, {v['seconds']} s)"} for k, v in r.RIGS.items()],
        "actions": opt(["auto"] + list(r.ACTIONS)), "lights": opt(r.LIGHTS), "aspects": opt(r.ASPECTS),
    }


def plan(p: dict):
    n = max(1, min(12, int(p.get("shots") or 4)))
    kw = dict(car=p.get("car") or "", rival=p.get("rival") or "", world=p.get("world") or "mountain pass", world_custom=p.get("world_custom") or "",
              subject_type=p.get("subject_type") or "car", style=p.get("style") or "photoreal", rig=p.get("rig") or "auto (race sequence)",
              action=p.get("action") or "auto", light=p.get("light") or "world default", intensity=float(p.get("intensity") or 0.85),
              aspect=p.get("aspect") or "16:9 landscape", seconds=float(p.get("seconds") or 0), extra=p.get("extra") or "", seed=int(p.get("seed") or 7))
    if kw["world_custom"]:
        kw["world"] = "custom"
    return [{k: s[k] for k in ("label", "rig", "action", "seconds", "motion_prompt")} for s in (rigs.compose(shot_index=i, **kw) for i in range(n))]
