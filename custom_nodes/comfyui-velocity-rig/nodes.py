"""Velocity Rig nodes for ComfyUI.

Velocity Rig        car + rival + world + camera rig -> first-frame prompt, LTX motion prompt, sizes, frames, seed, FX
Velocity Rig List   the whole rig/world/action library as text, to browse inside ComfyUI
Velocity FX         shake, speed blur, crash zoom, chroma, speed ramp on the decoded frames
Velocity Reel       stitch the last N saved shots into one hard-cut reel with their LTX audio
"""
from __future__ import annotations

import json

from . import rigs

CAT = "Velocity Rig"
RIG_CHOICES = ["auto (race sequence)", "auto (cycle by shot)", "random"] + list(rigs.RIGS)


class VelocityRig:
    CATEGORY = CAT
    FUNCTION = "run"
    DESCRIPTION = ("Pick a hero car, a rival car, a world and a camera rig; get the first-frame prompt and the LTX motion "
                   "prompt that shoot it like a high-speed motorsport commercial. 'auto (race sequence)' walks a curated "
                   "multi-car race arc (launch -> bumper draft -> hard brake -> wheel-cam hairpin -> FPV drift -> door-mount "
                   "overtake -> reverse bumper defense -> whip-pan finish).")
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT",
                    "INT", "STRING", "STRING")
    RETURN_NAMES = ("plate_prompt", "motion_prompt", "plate_negative", "video_negative", "plate_width", "plate_height",
                    "width", "height", "half_width", "half_height", "frames", "fps", "seed", "fx", "label")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_type": (list(rigs.ARCHETYPES), {"default": "car",
                                                        "tooltip": "Subject archetype: car (Supercars), motorcycle (GP Superbikes), jet (Fighter Jets), action_sports (Downhill MTB), speedboat (Powerboats)."}),
                "style": (list(rigs.STYLES), {"default": "photoreal",
                                              "tooltip": "Visual & animation style: photoreal (large-format cinema), anime (mountain touge), cartoon_3d (comic-book CG), synthwave_retro (80s night drive), cyberpunk_anime (neon city anime), comic_graphic_novel."}),
                "car": ("STRING", {"multiline": True, "default": "",
                                   "tooltip": "Describe your hero vehicle / subject (leave blank to use archetype default). Keep it consistent across shots."}),
                "world": (list(rigs.WORLDS) + ["custom"], {"default": "mountain pass"}),
                "rig": (RIG_CHOICES, {"default": "auto (race sequence)",
                                      "tooltip": "Camera rig: 'auto (race sequence)' follows the archetype's curated race spine; or pick any vehicle-mounted, chase, or trackside rig."}),
                "action": (["auto"] + list(rigs.ACTIONS), {"default": "auto"}),
                "light": (list(rigs.LIGHTS), {"default": "world default"}),
                "intensity": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Speed words in the prompt and strength of the rig's FX."}),
                "aspect": (list(rigs.ASPECTS), {"default": "16:9 landscape"}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 1.0,
                                      "tooltip": "0 = the rig's suggested length (2-4 s)."}),
                "shot_index": ("INT", {"default": 0, "min": 0, "max": 999, "control_after_generate": True,
                                       "tooltip": "Set to 'increment' and queue N times for N consecutive race shots."}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "rival": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "Rival vehicle description (leave blank to use archetype default, or 'none' for solo run)."}),
                "world_custom": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Your own location; overrides the world preset when filled."}),
                "extra": ("STRING", {"multiline": True, "default": "", "tooltip": "Appended to both prompts."}),
            },
        }

    def run(self, subject_type, car, world, rig, action, light, intensity, aspect, seconds, shot_index, seed,
            style="photoreal", rival="", world_custom="", extra=""):
        s = rigs.compose(subject_type=subject_type, style=style, car=car, rival=rival, world=world, rig=rig, world_custom=world_custom, light=light,
                         action=action, intensity=intensity, shot_index=shot_index, seed=seed, aspect=aspect,
                         seconds=seconds, extra=extra)
        return (s["plate_prompt"], s["motion_prompt"], s["plate_negative"], s["video_negative"], s["plate_width"],
                s["plate_height"], s["width"], s["height"], s["width"] // 2, s["height"] // 2, s["frames"],
                float(s["fps"]), s["seed"], json.dumps(s["fx"]), s["label"])


class VelocityRigList:
    CATEGORY = CAT
    FUNCTION = "run"
    DESCRIPTION = "Every camera rig, race sequence beat, world and action in the library, as text."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("library",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        lines = ["RACE SEQUENCE (auto default)"]
        for idx, (r_name, a_name) in enumerate(rigs.RACE_SEQUENCE, 1):
            lines.append(f"  {idx:02d}. {r_name} -> {a_name}")
        lines += ["", "CAMERA RIGS"]
        for name, r in rigs.RIGS.items():
            lines.append(f"- {name} [{rigs.GROUPS[r['group']]}, {r['seconds']} s]: {r['motion'].split('.')[0]}.")
        lines += ["", "WORLDS"] + [f"- {k}: {v['scene']}" for k, v in rigs.WORLDS.items()]
        lines += ["", "ACTIONS"] + [f"- {k}" for k in rigs.ACTIONS]
        return ("\n".join(lines),)


class VelocityFX:
    CATEGORY = CAT
    FUNCTION = "run"
    DESCRIPTION = ("Speed effects on decoded frames: rig shake, radial speed blur, crash zoom, chroma fringing and a "
                   "speed ramp. Connect the rig's fx output for its defaults; any slider >= 0 overrides. Frame count "
                   "is unchanged, so audio stays in sync.")
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        ov = lambda tip: ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": tip})
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Scales every effect. 0 = bypass."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "fx": ("STRING", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
                "shake": ov("-1 = rig default"),
                "speed_blur": ov("-1 = rig default"),
                "zoom_punch": ov("-1 = rig default"),
                "chroma": ov("-1 = rig default"),
                "ramp": ov("-1 = rig default"),
            },
        }

    def run(self, images, strength, seed, fx="", fps=24.0, **overrides):
        from . import fx as velocity_fx
        base = json.loads(fx) if fx else {}
        amt = {}
        for k in velocity_fx.KEYS:
            v = overrides.get(k, -1.0)
            amt[k] = (v if v is not None and v >= 0 else float(base.get(k, 0.0))) * strength
        if strength <= 0 or not any(amt.values()):
            return (images,)
        return (velocity_fx.apply(images, seed=seed, fps=fps or 24.0, **amt),)


class VelocityReel:
    CATEGORY = CAT
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Stitch the last N shots saved under a prefix (oldest first) into one reel: hard cuts, the first frames "
                   "of each shot trimmed, LTX audio kept. Queue your shots first, then run this on its own.")
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("reel", "shots")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prefix": ("STRING", {"default": "velocity_rig/shot", "tooltip": "SaveVideo filename_prefix of the shots."}),
            "count": ("INT", {"default": 6, "min": 1, "max": 64}),
            "trim_frames": ("INT", {"default": 6, "min": 0, "max": 48, "tooltip": "Frames cut from the start of each shot."}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")  # new shots may have landed since the last run

    def run(self, prefix, count, trim_frames, fps):
        import tempfile
        import time
        from pathlib import Path

        import folder_paths
        from comfy_api.input_impl import VideoFromFile

        from . import reel
        out = Path(folder_paths.get_output_directory())
        sub, stem = (out / prefix).parent, Path(prefix).name
        clips = sorted(sub.glob(f"{stem}_*.mp4"), key=lambda p: p.stat().st_mtime)[-count:]
        if not clips:
            raise FileNotFoundError(f"no shots matching {sub}/{stem}_*.mp4: queue some Velocity Rig shots first")
        dest = Path(tempfile.gettempdir()) / f"velocity_reel_{int(time.time())}.mp4"
        reel.assemble(clips, dest, fps=fps, trim_frames=trim_frames)
        return (VideoFromFile(str(dest)), "\n".join(c.name for c in clips))


class VelocityRigBuilder:
    CATEGORY = CAT
    FUNCTION = "run"
    DESCRIPTION = ("Interactive Camera Rig Builder: customize the exact mount anchor, camera pitch, lens focal length, "
                   "and mount physics stability for ANY subject archetype. Replaces fixed presets with an infinite camera-rig sandbox.")
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT",
                    "INT", "STRING", "STRING")
    RETURN_NAMES = ("plate_prompt", "motion_prompt", "plate_negative", "video_negative", "plate_width", "plate_height",
                    "width", "height", "half_width", "half_height", "frames", "fps", "seed", "fx", "label")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_type": (list(rigs.ARCHETYPES), {"default": "car",
                                                        "tooltip": "Subject archetype: car, motorcycle, jet, action_sports, speedboat."}),
                "style": (list(rigs.STYLES), {"default": "photoreal",
                                              "tooltip": "Animation / visual style: photoreal, anime (mountain touge), cartoon_3d (comic-book CG), synthwave_retro (80s night drive), cyberpunk_anime, comic_graphic_novel."}),
                "mount_anchor": (list(rigs.CUSTOM_ANCHORS), {"default": "front_splitter",
                                                             "tooltip": "Physical mounting position on or around the vehicle."}),
                "camera_pitch": (list(rigs.CUSTOM_PITCHES), {"default": "level",
                                                             "tooltip": "Camera horizon roll & pitch: level, low tilt-up hero, high tilt-down, canted Dutch tilt, or reverse-facing."}),
                "lens_profile": (list(rigs.CUSTOM_LENSES), {"default": "24mm cinematic wide",
                                                             "tooltip": "Optical focal length & lens character: 12mm fisheye, 24mm wide, 35mm doc, 50mm anamorphic, 85mm portrait, 135mm telephoto, macro."}),
                "mount_stability": (list(rigs.CUSTOM_STABILITY), {"default": "rigid_bolt",
                                                                 "tooltip": "Mount physics: rigid chassis bolt (max vibration), suction damped, 3-axis gyro gimbal, or raw handheld."}),
                "action": (["auto"] + list(rigs.ACTIONS), {"default": "auto"}),
                "world": (list(rigs.WORLDS) + ["custom"], {"default": "mountain pass"}),
                "light": (list(rigs.LIGHTS), {"default": "world default"}),
                "intensity": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect": (list(rigs.ASPECTS), {"default": "16:9 landscape"}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "shot_index": ("INT", {"default": 0, "min": 0, "max": 999, "control_after_generate": True}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "car": ("STRING", {"multiline": True, "default": "", "tooltip": "Leave blank to use archetype default."}),
                "rival": ("STRING", {"multiline": True, "default": "", "tooltip": "Leave blank for archetype default, or 'none' for solo."}),
                "world_custom": ("STRING", {"multiline": True, "default": ""}),
                "extra": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def run(self, subject_type, mount_anchor, camera_pitch, lens_profile, mount_stability,
            action, world, light, intensity, aspect, seconds, shot_index, seed,
            style="photoreal", car="", rival="", world_custom="", extra=""):
        s = rigs.compose_custom_rig(
            subject_type=subject_type, style=style, mount_anchor=mount_anchor, camera_pitch=camera_pitch,
            lens_profile=lens_profile, mount_stability=mount_stability, action=action,
            car=car, rival=rival, world=world, world_custom=world_custom, light=light,
            intensity=intensity, shot_index=shot_index, seed=seed, aspect=aspect,
            seconds=seconds, extra=extra)
        return (s["plate_prompt"], s["motion_prompt"], s["plate_negative"], s["video_negative"], s["plate_width"],
                s["plate_height"], s["width"], s["height"], s["width"] // 2, s["height"] // 2, s["frames"],
                float(s["fps"]), s["seed"], json.dumps(s["fx"]), s["label"])


NODE_CLASS_MAPPINGS = {
    "VelocityRig": VelocityRig,
    "VelocityRigBuilder": VelocityRigBuilder,
    "VelocityRigList": VelocityRigList,
    "VelocityFX": VelocityFX,
    "VelocityReel": VelocityReel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VelocityRig": "Velocity Rig (preset rig)",
    "VelocityRigBuilder": "Velocity Rig Builder (custom mount & lens)",
    "VelocityRigList": "Velocity Rig Library",
    "VelocityFX": "Velocity FX (speed)",
    "VelocityReel": "Velocity Reel (stitch shots)",
}
