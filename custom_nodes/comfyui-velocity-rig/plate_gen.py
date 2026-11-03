"""Off-GPU camera-rig plate & car-reference generator using Codex and Nano Banana (Gemini).

Generates consistent Hero + Rival car reference sheets and camera-rig first frames without touching
the Mac Studio GPU, leaving 100% of VRAM/MPS free for LTX-2.5 video + audio generation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

CODEX_BIN = "/opt/homebrew/bin/codex"
NANO_SCRIPT = "/Users/jefe/projects/jk-creative-factory/scripts/gemini-web-still.mjs"
NODE_BIN = "/opt/homebrew/bin/node"


def _fit_and_save(src: Path, dest: Path, width: int, height: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        if rgb.size != (width, height):
            rgb = ImageOps.fit(rgb, (width, height), method=Image.Resampling.LANCZOS)
        rgb.save(dest, format="PNG")
    return dest


def generate_codex(prompt: str, dest: Path, width: int = 1024, height: int = 576,
                   refs: list[Path] | None = None, pilot_ref: Path | None = None, timeout: int = 240) -> Path:
    """Generate or edit a plate using Codex CLI's built-in image generation tool."""
    refs = [Path(r).resolve() for r in (refs or []) if r and Path(r).exists()]
    orient = "landscape 16:9 (1536x1024)" if width >= height else "portrait 9:16 (1024x1536)"
    with tempfile.TemporaryDirectory(prefix="vr_codex_") as tmp:
        tmpdir = Path(tmp)
        local_refs = []
        for idx, r in enumerate(refs, 1):
            lr = tmpdir / f"ref_{idx}.png"
            shutil.copy(r, lr)
            local_refs.append(lr)

        ref_note = ""
        if len(local_refs) == 1:
            ref_note = ("Use the attached reference image (ref_1.png) as the exact visual identity for the hero car "
                        "(keep the exact same car body design, paint color, racing number, stripe and wheels). ")
        elif len(local_refs) >= 2:
            ref_note = ("Use the first attached image (ref_1.png) as the exact visual identity for our hero car, "
                        "and the second attached image (ref_2.png) as the exact visual identity for the rival car "
                        "(keep their exact body shapes, paint colors, racing numbers and wheels). ")

        if pilot_ref and Path(pilot_ref).exists():
            p_ref = tmpdir / "pilot_ref.png"
            shutil.copy(Path(pilot_ref), p_ref)
            local_refs.append(p_ref)
            ref_note += ("CRITICAL REQUIREMENT: The attached image (pilot_ref.png) is the REAL PERSON who MUST be depicted as the human pilot / driver in the cockpit! "
                         "Keep their EXACT facial features, face shape, eyes, skin tone, and hair from pilot_ref.png, showing them seated in the cockpit wearing a high-performance fighter pilot helmet, flight suit, and tactical gear. "
                         "Their face must be clearly and recognizably the person from pilot_ref.png! ")

        out_raw = tmpdir / "out.png"
        instruction = (
            f"Use your built-in image generation tool to generate ONE {orient} image matching the exact visual style, action, and camera perspective specified below. "
            f"{ref_note}Scene and camera rig specification:\n\n{prompt}\n\n"
            f"After generating the image, copy the generated PNG file to {out_raw}. "
            f"Do not create any other files. Reply only with {out_raw}."
        )

        env = os.environ.copy()
        env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
        env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"

        cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-C", str(tmpdir)]
        for lr in local_refs:
            cmd += ["-i", str(lr)]
        cmd += ["--", instruction]

        subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)

        if not out_raw.exists():
            pngs = [p for p in tmpdir.glob("*.png") if not p.name.startswith("ref_")]
            if pngs:
                out_raw = max(pngs, key=lambda p: p.stat().st_mtime)
        if not out_raw.exists():
            raise RuntimeError(f"Codex image generation did not produce an image for {dest.name}")
        return _fit_and_save(out_raw, dest, width, height)


def generate_nano_banana(prompt: str, dest: Path, width: int = 1024, height: int = 576,
                         refs: list[Path] | None = None, pilot_ref: Path | None = None, timeout: int = 180) -> Path:
    """Generate a plate using the Nano Banana (Gemini Create Image) harness."""
    refs = [Path(r).resolve() for r in (refs or []) if r and Path(r).exists()]
    with tempfile.TemporaryDirectory(prefix="vr_nano_") as tmp:
        out_raw = Path(tmp) / "out.png"
        ref_note = ""
        if len(refs) == 1:
            ref_note = "Match the exact hero car design, color, number and wheels from the first reference image. "
        elif len(refs) >= 2:
            ref_note = ("Match the exact hero car from the first reference image and the exact rival car from the "
                        "second reference image. ")
        aspect = "16:9 widescreen" if width >= height else "9:16 vertical"
        if pilot_ref and Path(pilot_ref).exists():
            ref_note += "CRITICAL: Depict the exact person from the pilot reference image sitting in the cockpit as the pilot with their exact facial likeness and hair. "
            refs.append(Path(pilot_ref))
        full_prompt = f"{ref_note}{prompt} Format: {aspect}, single unbroken frame, no text or watermarks."

        env = os.environ.copy()
        env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
        cmd = [NODE_BIN, NANO_SCRIPT, "--prompt", full_prompt, "--out", str(out_raw)]
        for r in refs:
            cmd += ["--ref", str(r)]

        subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)
        if not out_raw.exists():
            raise RuntimeError(f"Nano Banana did not produce an image for {dest.name}")
        return _fit_and_save(out_raw, dest, width, height)


def generate_plate(prompt: str, dest: Path, width: int = 1024, height: int = 576,
                   refs: list[Path] | None = None, pilot_ref: Path | None = None, backend: str = "codex") -> Path:
    """Generate a shot plate or car reference using the chosen backend ('codex', 'nano_banana', or 'auto')."""
    if dest.exists():
        return dest
    if backend == "nano_banana":
        try:
            return generate_nano_banana(prompt, dest, width=width, height=height, refs=refs, pilot_ref=pilot_ref)
        except Exception:
            return generate_codex(prompt, dest, width=width, height=height, refs=refs, pilot_ref=pilot_ref)
    return generate_codex(prompt, dest, width=width, height=height, refs=refs, pilot_ref=pilot_ref)


def ensure_car_ref(car_desc: str, dest: Path, backend: str = "codex", style_desc: str = "") -> Path:
    """Generate a clean 3/4 pit-lane identity reference sheet for a vehicle in the chosen visual style."""
    if dest.exists():
        return dest
    return generate_plate(car_ref_prompt(car_desc, style_desc), dest, width=1024, height=576, refs=None, backend=backend)


def car_ref_prompt(car_desc: str, style_desc: str = "", airborne: bool = False) -> str:
    style_header = style_desc or "professional motorsport studio three-quarter front-and-side reference photograph"
    where = ("flying level in mid-air against a plain soft grey sky with the landing gear fully retracted"
             if airborne else "parked on a clean studio floor")
    return (
        f"Clean {style_header} of {car_desc}, "
        f"{where} under soft diffused lighting so the full vehicle silhouette, paint finish, "
        f"racing number, aero components and wheel design are crystal clear. "
        f"Original fictional vehicle design with no real brand badges, no logos and no text."
    )
