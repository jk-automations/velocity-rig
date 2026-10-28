"""Velocity Rig batch runner: N fast-paced camera-rig shots on open weights, end to end. Z-Image Turbo draws each identity ref,
Qwen Image Edit 2511 locks every plate to them, LTX-2.5 22B distilled makes video + audio, Velocity FX adds the rig physics,
then the shots are stitched into a reel. Every render goes through the Comfy SDK (Comfy API v2).

    python -m velocityrig --world "mountain pass" --shots 6 --seed 7 --out runs/velocity-rig/race1
        [--car "a gloss teal wedge coupe #27..."] [--rival "a gloss crimson widebody fastback #14..."]
        [--photo my_car.png] [--rival-photo rival.png] [--pilot-photo me.png] [--rig "auto (race sequence)"]
        [--light "golden hour"] [--intensity 0.85] [--aspect "16:9 landscape"] [--plates-only] [--no-one-prompt] [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from velocityrig import comfy
from velocityrig.graphs.edit import qwen_edit
from velocityrig.graphs.still import text_to_image
from velocityrig.graphs.video import ltx_i2v

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "custom_nodes" / "comfyui-velocity-rig"
TRIM_FRAMES = 6
CLIP_CHUNK = 3  # LTX clips per ComfyUI prompt when --no-one-prompt


def _load(name):
    spec = importlib.util.spec_from_file_location(f"velocity_rig_{name}", PACK / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rigs, reel_lib, fx_lib = (_load(n) for n in ("rigs", "reel", "fx"))
_log_lock = threading.Lock()
T0 = time.time()


def log(msg, step=False):
    with _log_lock:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
        if step:
            comfy.emit("step", msg=msg)


def apply_fx_to_video(raw_mp4: Path, dest_mp4: Path, fx_params: dict, fps: float, seed: int) -> Path:
    """Decode raw_mp4 frames, apply VelocityFX (shake, radial speed blur, zoom punch, chroma, ramp), and mux the LTX audio back."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    with tempfile.TemporaryDirectory(prefix="vr_fx_") as tmp:
        in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_mp4), str(in_dir / "%05d.png")], check=True)
        files = sorted(in_dir.glob("*.png"))
        if not files:
            shutil.copy(raw_mp4, dest_mp4)
            return dest_mp4
        tensor = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in files]))
        out = fx_lib.apply(tensor, seed=seed, fps=fps, **{k: float(fx_params.get(k, 0.0)) for k in fx_lib.KEYS})
        for idx, arr in enumerate((out.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8), 1):
            Image.fromarray(arr).save(out_dir / f"{idx:05d}.png")
        dest_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(int(fps)), "-i", str(out_dir / "%05d.png"),
                        "-i", str(raw_mp4), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "17", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(dest_mp4)], check=True)
    return dest_mp4


def make_plate(shot, i, out: Path, ref: Path, worker: str | None, pilot_ref: Path | None = None, defer: list | None = None) -> Path:
    """Qwen Image Edit renders the plate from the hero ref only: given two vehicle refs it fuses them into one, so the rival
    comes from the text. With `defer` the graph is queued for comfy.run_many instead of run (and nothing is logged)."""
    dest = out / "plates" / f"{i + 1:02d}.png"
    if dest.exists():
        if defer is None:
            log(f"plate {i + 1:02d} {shot['rig']} / {shot['action']} ready", True)
        return dest
    t = time.time()
    ref_imgs = [ref] + ([pilot_ref] if pilot_ref and pilot_ref.exists() else [])
    lead = ("Image 1 shows only the design of the hero vehicle: copy its exact body shape, paint, markings and number, "
            "but not its pose, background or camera angle. ")
    if len(ref_imgs) > 1:
        lead += "Image 2 is the pilot: keep their exact face. "
    prompt = shot["plate_prompt"] + " " + lead + "Any other vehicle in the scene is a different design, as described."
    g = qwen_edit(prompt, len(ref_imgs), seed=shot["seed"], width=shot["plate_width"], height=shot["plate_height"],
                  negative=shot["plate_negative"], prefix="velocity_rig/plate")
    comfy.run_or_defer(defer, g, {"save": dest}, url=worker, label=f"plate {i + 1}", inputs={(f"img{n}", "image"): r for n, r in enumerate(ref_imgs, 1)})
    if defer is None:
        log(f"plate {i + 1:02d} {shot['rig']} / {shot['action']} (Qwen Image Edit) in {time.time() - t:.0f}s", True)
    return dest


def make_clip(shot, i, plate: Path, out: Path, worker: str | None, defer: list | None = None) -> Path:
    """LTX-2.5 + Velocity FX. With `defer` only the LTX graph is queued; call again after run_many to apply the FX."""
    dest, raw = out / "clips" / f"{i + 1:02d}_shot.mp4", out / "clips" / f"{i + 1:02d}_raw.mp4"
    if dest.exists():
        if defer is None:
            log(f"clip  {i + 1:02d} {shot['rig']} / {shot['action']} cached", True)
        return dest
    t = time.time()
    if not raw.exists():
        g = ltx_i2v(shot["motion_prompt"], width=shot["width"], height=shot["height"], seconds=int(shot["seconds"]), fps=int(shot["fps"]),
                    seed=shot["seed"], negative=shot["video_negative"], prefix="velocity_rig/raw", audio=True)
        comfy.run_or_defer(defer, g, {"save": raw}, url=worker, label=f"shot {i + 1} {shot['rig']}", inputs={("img1", "image"): plate})
        if defer is not None:
            return dest
    apply_fx_to_video(raw, dest, shot["fx"], fps=shot["fps"], seed=shot["seed"])
    log(f"clip  {i + 1:02d} {shot['rig']} / {shot['action']} (LTX-2.5 + Velocity FX) in {time.time() - t:.0f}s", True)
    return dest


def build_contact_sheet(plates: list[Path], shots: list[dict], refs: list[Path], dest: Path):
    items = []
    for ref, tag, key in zip(refs, ("HERO", "RIVAL"), ("car", "rival")):
        if ref.exists():
            items.append((ref, f"REF {tag}: {(shots[0].get(key, '') if shots else '')[:42].upper()}"))
    items += [(p, s["label"]) for p, s in zip(plates, shots) if p.exists()]
    if not items:
        return
    cols, tw, th, pad, lh = 2, 512, 288, 16, 30
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lh + pad) + pad), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    for idx, (p, title) in enumerate(items):
        r, c = divmod(idx, cols)
        x, y = pad + c * (tw + pad), pad + r * (th + lh + pad)
        with Image.open(p) as im:
            sheet.paste(im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS), (x, y + lh))
        draw.text((x + 4, y + 6), title, fill=(235, 220, 140))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--subject-type", default="car", choices=list(rigs.ARCHETYPES), help="car, motorcycle, jet, jet_pilot_pov, action_sports, speedboat")
    ap.add_argument("--style", default="photoreal", choices=list(rigs.STYLES))
    ap.add_argument("--car", default="", help="hero vehicle description (blank = the archetype's default)")
    ap.add_argument("--rival", default="", help="rival description (blank = default, 'none' = solo)")
    ap.add_argument("--photo", help="photo of your own vehicle: becomes the hero identity ref")
    ap.add_argument("--rival-photo")
    ap.add_argument("--pilot-photo", help="photo of yourself, placed in the cockpit as the pilot")
    ap.add_argument("--pilot-name", default="")
    ap.add_argument("--world", default="mountain pass")
    ap.add_argument("--world-custom", default="")
    ap.add_argument("--rig", default="auto (race sequence)", help="a rig name, 'auto (race sequence)', 'auto (cycle by shot)', 'random', or a comma list")
    ap.add_argument("--action", default="auto")
    ap.add_argument("--light", default="world default")
    ap.add_argument("--intensity", type=float, default=0.85)
    ap.add_argument("--seconds", type=float, default=0.0, help="per-shot seconds (0 = the rig's own 2-4 s)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--aspect", default="16:9 landscape")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plates-only", action="store_true")
    ap.add_argument("--one-prompt", action=argparse.BooleanOptionalAction, default=True,
                    help="chain each plate into its LTX graph and merge the whole reel into one ComfyUI prompt (one GPU-queue turn); --no-one-prompt batches plates and clips separately")
    ap.add_argument("--comfy", default=None, help="Comfy API v2 base URL (default: $COMFY_BASE_URL or http://127.0.0.1:8195)")
    ap.add_argument("--dry-run", action="store_true", help="build and save the real graphs, write placeholder outputs, contact no server")
    a = ap.parse_args(argv)
    if a.dry_run:
        comfy.DRY = True

    out = Path(a.out)
    for d in ("refs", "plates", "clips"):
        (out / d).mkdir(parents=True, exist_ok=True)
    photo = Path(a.photo).resolve() if a.photo else None
    rival_photo = Path(a.rival_photo).resolve() if a.rival_photo else None
    pilot_ref = None
    if a.pilot_photo:
        pilot_ref = out / "refs" / "pilot.png"
        if not pilot_ref.exists():
            shutil.copy(Path(a.pilot_photo).resolve(), pilot_ref)
    car = a.car or ("the car in image 1" if photo else "")
    rig_list = [r.strip() for r in a.rig.split(",")] if "," in a.rig else [a.rig] * a.shots

    shots = [rigs.compose(car=car, rival=a.rival, world=a.world if not a.world_custom else "custom", world_custom=a.world_custom,
                          subject_type=a.subject_type, style=a.style, rig=rig_list[i % len(rig_list)], action=a.action, light=a.light,
                          intensity=a.intensity, aspect=a.aspect, seconds=a.seconds, extra=a.extra, shot_index=i, seed=a.seed) for i in range(a.shots)]
    (out / "shots.json").write_text(json.dumps(shots, indent=1))
    wanted = [("hero", photo, shots[0]["car"])] + ([("rival", rival_photo, shots[0]["rival"])] if shots[0].get("rival") else [])
    comfy.emit("plan", total=len(wanted) + a.shots * (1 if a.plates_only else 2) + (0 if a.plates_only else 1))
    log(f"{a.shots} shots: " + ", ".join(f"{s['rig']} ({s['action']})" for s in shots))

    style_desc = rigs.STYLES[a.style]["label"]

    def make_ref(n, name, src, desc):
        dest = out / "refs" / f"{name}.png"
        if src and not dest.exists():
            shutil.copy(src, dest)
        if not dest.exists():
            t = time.time()
            g = text_to_image(rigs.car_ref_prompt(desc, style_desc, airborne=a.subject_type.startswith("jet")), width=1024, height=576,
                              seed=a.seed + n, prefix=f"velocity_rig/ref_{name}")
            comfy.run(g, {"10": dest}, url=a.comfy, label=f"{name} ref")
            log(f"reference {name} (Z-Image Turbo) in {time.time() - t:.0f}s", True)
        else:
            log(f"reference {name} ready", True)
        return dest

    with ThreadPoolExecutor(len(wanted)) as pool:
        refs = [f.result() for f in [pool.submit(make_ref, n, *w) for n, w in enumerate(wanted)]]

    plate_of = lambda i: out / "plates" / f"{i + 1:02d}.png"
    if a.one_prompt and not a.plates_only:
        jobs = []
        for i, sh in enumerate(shots):
            pj, cj = [], []
            make_plate(sh, i, out, refs[0], a.comfy, pilot_ref, defer=pj)
            make_clip(sh, i, plate_of(i), out, a.comfy, defer=cj)
            if pj and cj:  # plate -> LTX inside one graph
                jobs.append(dict(graph=comfy.chain(pj[0]["graph"], "dec", cj[0]["graph"], "img1"), url=a.comfy, label=f"shot {i + 1} plate+clip",
                                 outputs={"u_save": pj[0]["outputs"]["save"], "d_save": cj[0]["outputs"]["save"]},
                                 inputs={("u_" + n, k): v for (n, k), v in pj[0]["inputs"].items()}))
            else:
                jobs += pj + cj
        if jobs:
            log(f"rendering {len(shots)} shots (plate + LTX) in one ComfyUI prompt")
            comfy.run_many(jobs, chunk=len(jobs))
    else:
        batch: list = []
        for i, sh in enumerate(shots):
            make_plate(sh, i, out, refs[0], a.comfy, pilot_ref, defer=batch)
        if batch:
            log(f"rendering {len(batch)} plates in one ComfyUI prompt")
            comfy.run_many(batch, chunk=len(batch))
    plates = [make_plate(sh, i, out, refs[0], a.comfy, pilot_ref) for i, sh in enumerate(shots)]
    sheet = out / "plates" / "contact_sheet.jpg"
    build_contact_sheet(plates, shots, refs, sheet)
    if a.plates_only:
        log(f"plates_only complete: {sheet}")
        return

    if not a.one_prompt:
        batch = []
        for i, (sh, p) in enumerate(zip(shots, plates)):
            make_clip(sh, i, p, out, a.comfy, defer=batch)
        if batch:
            log(f"rendering {len(batch)} LTX clips, {CLIP_CHUNK} per ComfyUI prompt")
            comfy.run_many(batch, chunk=CLIP_CHUNK)
    clips = [make_clip(sh, i, p, out, a.comfy) for i, (sh, p) in enumerate(zip(shots, plates))]
    reel = out / "reel.mp4"
    reel_lib.assemble(clips, reel, shots[0]["fps"], TRIM_FRAMES)
    log(f"reel: {reel} ({len(clips)} shots)", True)


if __name__ == "__main__":
    main(sys.argv[1:])
