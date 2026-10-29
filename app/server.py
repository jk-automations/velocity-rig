"""Rig app: FastAPI jobs queue + SSE progress + executed-ComfyUI-graph viewer. Entry-specific bits live in appcfg.py.

    python app/server.py [--port 8811] [--dry-run]

--dry-run (or RIG_DRY=1) runs the runner with --dry-run: real graphs are built and saved, outputs are placeholders, no
ComfyUI is contacted. Runs land in runs/app/<id>; any CLI run under runs/ shows up in the gallery too.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path[:0] = [str(APP), str(ROOT)]

import appcfg as C  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

RUNS = Path(os.environ.get("RIG_RUNS") or ROOT / "runs").resolve()
RUNS.mkdir(parents=True, exist_ok=True)
DRY = os.environ.get("RIG_DRY") == "1"
REPLAY: Path | None = None  # --replay-from <finished run dir>: copy it into each new job and run the REAL runner on the cache
MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov", ".gif"}
UP_OK = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".mkv", ".m4v"}
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
Q: "queue.Queue[str]" = queue.Queue()
PROCS: dict[str, subprocess.Popen] = {}


@asynccontextmanager
async def lifespan(_app):
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title=C.CONFIG["name"], lifespan=lifespan)


def run_dir(rel: str) -> Path:
    p = (RUNS / rel).resolve()
    if RUNS not in p.parents or not p.is_dir():
        raise HTTPException(404, "no such run")
    return p


def rel(p: Path) -> str:
    return p.relative_to(RUNS).as_posix()


def touch(job: dict, force: bool = False):
    job["rev"] = job.get("rev", 0) + 1
    if force or job["status"] != "running" or job["rev"] % 20 == 0:
        try:
            (RUNS / job["run"] / "job.json").write_text(json.dumps({k: v for k, v in job.items() if k != "log"}, default=str))
        except OSError:
            pass


def default_args(params: dict, uploads: dict) -> list[str]:
    out: list[str] = []
    for c in C.CONFIG["controls"]:
        v = params.get(c["id"], c.get("default"))
        if not c.get("flag") or v is None or v == "" or v is False:
            continue
        out += [c["flag"]] if c["type"] == "toggle" else [c["flag"], str(v)]
    for u in C.CONFIG["uploads"]:
        if u["id"] in uploads:
            out += [u["flag"], str(uploads[u["id"]])]
    return out


def run_job(job: dict):
    out = RUNS / job["run"]
    args = (C.build_args if hasattr(C, "build_args") else default_args)(job["params"], job["uploads"])
    cmd = [sys.executable, "-u", "-m", C.MODULE, *args, "--out", str(out)] + (["--dry-run"] if DRY else [])
    env = dict(os.environ, PYTHONUNBUFFERED="1", RIG_EVENTS="1", **({"RIG_DRY": "1"} if DRY else {}))
    if REPLAY:
        # Replay: seed the output dir with a finished render. The runner reuses every cached file; RIG_REPLAY makes the Comfy layer
        # refuse to submit anything, so a missing file aborts the job instead of touching a GPU.
        shutil.copytree(REPLAY, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("fx-full", "*.log", "job.json", "__pycache__"))
        env["RIG_REPLAY"] = "1"
        job["log"].append(f"replaying a finished render from {REPLAY.name}: cached files reused, nothing is submitted to ComfyUI")
    job.update(status="running", started=time.time(), cmd=" ".join(cmd))
    touch(job, True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    PROCS[job["run"]] = p
    for line in p.stdout:
        line = line.rstrip()
        with LOCK:
            if line.startswith("@@"):
                kind, _, body = line[2:].partition(" ")
                try:
                    d = json.loads(body)
                except ValueError:
                    continue
                if kind == "plan":
                    job["total"] = int(d.get("total", 0))
                elif kind == "step":
                    job["done"] += 1
                    job["stage"] = d.get("msg", "")
                elif kind == "graph":
                    job["live"] = {"graph": d.get("file"), "status": "queued", "label": d.get("label")}
                elif kind == "live":
                    job["live"] = {**(job.get("live") or {}), **d}
            else:
                job["log"].append(line)
                del job["log"][:-300]
            job["progress"] = min(0.99, job["done"] / job["total"]) if job["total"] else 0
            touch(job)
    rc = p.wait()
    PROCS.pop(job["run"], None)
    with LOCK:
        cancelled = job.get("cancel")
        job.update(status="cancelled" if cancelled else "done" if rc == 0 else "failed", ended=time.time(), live=None)
        if rc == 0:
            job["progress"] = 1.0
        elif not cancelled:
            job["error"] = "\n".join(job["log"][-8:])
        touch(job, True)


def worker():
    while True:
        jid = Q.get()
        job = JOBS.get(jid)
        if job and job["status"] == "queued":
            try:
                run_job(job)
            except Exception as e:
                job.update(status="failed", error=str(e), ended=time.time())
                touch(job, True)


def load_jobs():
    for f in RUNS.glob("app/*/job.json"):
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        if j.get("status") in ("running", "queued"):
            j.update(status="failed", error="server restarted during the job")
        j.setdefault("log", [])
        JOBS[j["run"]] = j


# ---- graphs and files -------------------------------------------------------------------------------------------

_gcache: dict = {}


def graph_models(graph: dict) -> list[str]:
    found: list[str] = []
    for n in graph.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, str):
                for pat, label in C.MODEL_PATTERNS:
                    if pat in v.lower() and label not in found:
                        found.append(label)
    return found


def graph_list(d: Path, job: dict | None) -> list[dict]:
    files = sorted(d.rglob("*.graph.json"), key=lambda f: (f.stat().st_mtime, f.name))
    out = []
    for i, f in enumerate(files, 1):
        key = (str(f), f.stat().st_mtime)
        if key not in _gcache:
            try:
                r = json.loads(f.read_text())
            except ValueError:
                continue
            g = r.get("graph") or {}
            _gcache[key] = {"label": r.get("label") or f.stem, "n_nodes": len(g), "models": graph_models(g), "job_id": r.get("job_id"),
                            "worker": r.get("worker"), "note": r.get("note")}
        status = "done"
        if job and job["status"] in ("running", "failed") and i == len(files):
            status = "running" if job["status"] == "running" else "failed"
        out.append({"idx": i, "file": f.relative_to(d).as_posix(), "status": status, **_gcache[key]})
    return out


def file_groups(d: Path) -> dict:
    g: dict = {"root": [], "refs": [], "plates": [], "clips": []}
    for f in sorted(d.rglob("*")):
        if f.suffix.lower() in MEDIA and "uploads" not in f.parts:
            top = f.relative_to(d).parts[0]
            g.setdefault(top if top in g else "root", []).append(f.relative_to(RUNS).as_posix())
    return g


def snapshot(job: dict | None, d: Path) -> dict:
    base = {"run": rel(d), "files": file_groups(d), "graphs": graph_list(d, job)}
    if job:
        base.update({k: v for k, v in job.items() if k not in ("uploads",)})
    else:
        base.update(status="done", params={}, progress=1.0, log=[], created=d.stat().st_mtime)
    return base


@app.get("/api/config")
def config():
    return {**C.CONFIG, "dry": DRY, "replay": REPLAY.name if REPLAY else None}


@app.get("/api/presets")
def presets():
    return C.library()


@app.post("/api/plan")
def plan(body: dict):
    try:
        return C.plan(body)
    except Exception as e:
        raise HTTPException(400, str(e))


_comfy: dict = {"t": 0}


@app.get("/api/comfy")
def comfy_status():
    if REPLAY:
        return {"up": False, "url": "none (replay)", "detail": "replay mode never contacts ComfyUI"}
    if time.time() - _comfy["t"] > 5:
        url = (os.environ.get("COMFY_BASE_URL") or "http://127.0.0.1:8195").rstrip("/")
        try:
            with urllib.request.urlopen(url + "/api/v2/health", timeout=1.5) as r:
                v = {"up": True, "detail": r.read(200).decode("utf8", "replace")}
        except Exception as e:
            v = {"up": False, "detail": str(e)[:120]}
        _comfy.update(t=time.time(), v={"url": url, **v})
    return _comfy["v"]


@app.post("/api/jobs")
async def create_job(params: str = Form("{}"), files: list[UploadFile] = File(default=[]), fields: str = Form("")):
    p = json.loads(params or "{}")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = RUNS / "app" / jid
    (d / "uploads").mkdir(parents=True)
    names = [n for n in fields.split(",") if n]
    uploads = {}
    for name, f in zip(names, files):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in UP_OK:
            raise HTTPException(400, f"unsupported upload type {ext or '?'}")
        dest = d / "uploads" / f"{name}{ext}"
        with dest.open("wb") as o:
            shutil.copyfileobj(f.file, o)
        uploads[name] = str(dest)
    if hasattr(C, "validate"):
        try:
            C.validate(p, uploads)
        except ValueError as e:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, str(e))
    job = {"id": jid, "run": rel(d), "status": "queued", "params": p, "uploads": uploads, "created": time.time(), "progress": 0.0,
           "done": 0, "total": 0, "stage": "queued", "log": [], "live": None}
    with LOCK:
        JOBS[job["run"]] = job
    touch(job, True)
    Q.put(job["run"])
    return {"run": job["run"]}


@app.get("/api/run")
def get_run(run: str):
    d = run_dir(run)
    return snapshot(JOBS.get(run), d)


@app.post("/api/cancel")
def cancel(run: str):
    job = JOBS.get(run)
    if not job or job["status"] not in ("queued", "running"):
        raise HTTPException(409, "not running")
    job["cancel"] = True
    p = PROCS.get(run)
    if p:
        os.killpg(p.pid, signal.SIGTERM)
    else:
        job["status"] = "cancelled"
    touch(job, True)
    return {"ok": True}


@app.get("/api/graph")
def get_graph(run: str, file: str):
    d = run_dir(run)
    f = (d / file).resolve()
    if d not in f.parents or not f.name.endswith(".graph.json") or not f.exists():
        raise HTTPException(404, "no such graph")
    return json.loads(f.read_text())


@app.get("/api/runs")
def runs():
    out = []
    for f in list(RUNS.glob("*/")) + list(RUNS.glob("*/*/")):
        if not f.is_dir() or f.name == "uploads":
            continue
        if not any((f / x).exists() for x in ("plates", "clips", "shots.json", "result.json", "job.json")) and not list(f.glob("*.graph.json")):
            continue
        fg = file_groups(f)
        sheet = next((x for x in fg["plates"] if x.endswith("contact_sheet.jpg")), None)
        reel = next((x for x in fg["root"] if x.endswith("reel.mp4")), None)
        poster = sheet or next((x for x in fg["root"] if x.endswith((".png", ".jpg")) and "report" not in x), None) or (fg["plates"] or [None])[0]
        job = JOBS.get(rel(f))
        out.append({"run": rel(f), "mtime": f.stat().st_mtime, "status": job["status"] if job else "done", "poster": poster, "reel": reel,
                    "videos": sum(x.endswith(".mp4") and not x.endswith("_raw.mp4") for g in fg.values() for x in g), "plates": len(fg["plates"]),
                    "title": (job or {}).get("params", {}).get("title") or f.name})
    return sorted(out, key=lambda r: -r["mtime"])[:80]


@app.get("/api/events")
async def events(run: str):
    d = run_dir(run)

    async def gen():
        last, idle = -1, 0.0
        while True:
            job = JOBS.get(run)
            rev = (job or {}).get("rev", 0)
            if rev != last:
                last = rev
                snap = snapshot(job, d)
                snap["queue_position"] = queue_pos(run)
                yield f"event: job\ndata: {json.dumps(snap, default=str)}\n\n"
                idle = 0
                if not job or job["status"] in ("done", "failed", "cancelled"):
                    break
            else:
                idle += 0.4
                if idle >= 12:
                    idle = 0
                    yield ": keepalive\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def queue_pos(run: str):
    with LOCK:
        queued = sorted((j for j in JOBS.values() if j["status"] == "queued"), key=lambda j: j["created"])
        running = any(j["status"] == "running" for j in JOBS.values())
    ids = [j["run"] for j in queued]
    return ids.index(run) + (1 if running else 0) if run in ids else None


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith(("/api/", "/runs/")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/runs", StaticFiles(directory=str(RUNS)), name="runs")
app.mount("/", StaticFiles(directory=str(APP / "web"), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=C.CONFIG.get("port", 8811))
    ap.add_argument("--dry-run", action="store_true", help="placeholder outputs, no ComfyUI")
    ap.add_argument("--replay-from", help="a finished run dir: every Render copies it and re-runs the real runner on the cache (no GPU)")
    a = ap.parse_args()
    if a.replay_from:
        REPLAY = Path(a.replay_from).expanduser().resolve()
        if not (REPLAY / "shots.json").exists():
            sys.exit(f"--replay-from: {REPLAY} has no shots.json")
    if a.dry_run:
        os.environ["RIG_DRY"] = "1"
        DRY = True
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
