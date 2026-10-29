"""Thin layer over the Comfy SDK: one client per ComfyUI worker, graph in, files out.

Every render goes through comfy_sdk -> Comfy API v2 (Comfy Developer Platform). Locally that is comfy-api-proxy in front of
ComfyUI (ops/serve.sh); point COMFY_BASE_URL at Comfy Cloud or a serverless deployment instead and nothing else changes.

RIG_EVENTS=1 makes run() print "@@graph" / "@@live" lines (the web app reads them for the live node view).
Renders can be merged into one ComfyUI prompt (run_many / chain), so a shared GPU queue hands out one turn per reel instead of one per render.
RIG_DRY=1 (runner --dry-run) skips the SDK: the real graph is still built and saved, outputs are placeholders.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8195"
DRY = os.environ.get("RIG_DRY") == "1"
_lock = threading.Lock()
_clients: dict = {}
RETRIES = 2
RETRY_WAIT_S = 30
_LOST = re.compile(r"reach ComfyUI|connect|interrupt|unknown job", re.I)
_FINAL = ("succeeded", "failed", "completed", "cancelled")


def emit(kind: str, **data):
    if os.environ.get("RIG_EVENTS") == "1":
        print(f"@@{kind} {json.dumps(data, default=str)}", flush=True)


def worker() -> str:
    return (os.environ.get("COMFY_BASE_URL") or DEFAULT_URL).strip().rstrip("/")


def client(url: str | None = None):
    url = (url or worker()).rstrip("/")
    with _lock:
        if url not in _clients:
            from comfy_sdk import Comfy
            prev = os.environ.get("COMFY_BASE_URL")  # the SDK reads its base URL from the environment when constructed
            os.environ["COMFY_BASE_URL"] = url
            try:
                _clients[url] = Comfy(timeout=120.0, client_info="comfy-rigs/0.1")
            finally:
                if prev is None:
                    os.environ.pop("COMFY_BASE_URL", None)
                else:
                    os.environ["COMFY_BASE_URL"] = prev
        return _clients[url]


def _follow(job, graph: dict, gfile: Path):
    try:
        for ev in job.events():
            kind = type(ev).__name__
            if kind == "StatusChange":
                emit("live", graph=str(gfile), status=ev.status, queue=ev.queue_position)
                if ev.status in _FINAL:
                    return
            elif kind == "Progress":
                node = str(ev.current_node) if ev.current_node is not None else None
                emit("live", graph=str(gfile), status="running", node=node, cls=(graph.get(node) or {}).get("class_type"),
                     step=ev.step, steps=ev.steps, nodes_done=ev.nodes_done, nodes_total=ev.nodes_total)
    except Exception:  # events are best-effort; the job itself is unaffected
        pass


def _submit_and_wait(c, graph, inputs, timeout, record, gfile):
    wf = c.workflows.from_json(graph)
    for (node_id, name), path in (inputs or {}).items():
        wf.set_input(node_id, name, c.assets.from_file(str(path)))
    job = c.submit(wf)
    record["job_id"] = job.id
    gfile.write_text(json.dumps(record, indent=1, default=str))
    if os.environ.get("RIG_EVENTS") == "1":
        threading.Thread(target=_follow, args=(job, record["graph"], gfile), daemon=True).start()
    job.wait(timeout=timeout)
    return job


def _dims(graph: dict, video: bool, default=(1024, 576)):
    want = "EmptyLTXVLatentVideo" if video else "EmptySD3LatentImage"
    for n in graph.values():
        if n.get("class_type") == want:
            i = n["inputs"]
            return (int(i["width"]) * 2, int(i["height"]) * 2) if video else (int(i["width"]), int(i["height"]))
    return default


def _placeholder(dest: Path, graph: dict, label: str):
    video = dest.suffix.lower() in (".mp4", ".mov", ".webm")
    w, h = _dims(graph, video)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if video:
        ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={w - w % 2}x{h - h % 2}:rate=24:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=220:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(dest)], check=True)
        return
    import numpy as np
    from PIL import Image, ImageDraw
    seed = sum(map(ord, label)) % 255
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    arr = np.stack([40 + seed // 4 + 90 * xs, 50 + 110 * ys, 100 + seed // 3 - 40 * xs], -1).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    ImageDraw.Draw(im).text((24, 24), f"DRY RUN  {label}", fill=(255, 255, 255))
    im.save(dest)


def _dry(graph, outputs, record, gfile, label):
    record["job_id"] = "dry-run"
    gfile.write_text(json.dumps(record, indent=1, default=str))
    ids = list(graph)
    for i, nid in enumerate(ids, 1):
        emit("live", graph=str(gfile), status="running", node=nid, cls=graph[nid].get("class_type"), step=1, steps=1,
             nodes_done=i - 1, nodes_total=len(ids))
        time.sleep(0.03)
    for dest in outputs.values():
        if Path(dest).suffix:
            _placeholder(Path(dest), graph, label)
    emit("live", graph=str(gfile), status="succeeded")
    return {nid: [Path(d)] for nid, d in outputs.items()}


def run(graph: dict, outputs: dict, *, url: str | None = None, timeout: float = 7200, label: str = "",
        inputs: dict | None = None) -> dict:
    """Run an API-format graph and save each listed output node's files.

    outputs maps node id -> destination. A destination with a suffix is a file path (first output only); otherwise it is a
    directory that receives every output file. inputs maps (node_id, input_name) -> local file uploaded as an asset.
    The submitted graph is saved beside the first output as <name>.graph.json (graph, ComfyUI job id, label) so every
    render can be reopened in ComfyUI and is shown in the app's graph viewer. Dropped jobs are resubmitted RETRIES times.
    """
    if os.environ.get("RIG_REPLAY") == "1":  # replay mode: never submit; a missing cached file aborts the job
        raise SystemExit(f"replay: {label or 'a render'} is not in the replayed run, refusing to submit anything to ComfyUI")
    first = Path(next(iter(outputs.values())))
    gfile = first.with_name(first.stem + ".graph.json") if first.suffix else first / f"{label or 'render'}.graph.json"
    gfile.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot before set_input: the SDK swaps asset objects into the graph dict in place.
    record = {"label": label, "worker": "dry-run" if DRY else (url or worker()), "graph": copy.deepcopy(graph),
              "inputs": {f"{n}.{k}": str(p) for (n, k), p in (inputs or {}).items()}}
    gfile.write_text(json.dumps(record, indent=1, default=str))
    emit("graph", file=str(gfile), label=label)
    if DRY:
        return _dry(graph, outputs, record, gfile, label)
    c = client(url)
    for attempt in range(1, RETRIES + 2):
        try:
            job = _submit_and_wait(c, copy.deepcopy(record["graph"]), inputs, timeout, record, gfile)
        except Exception as e:  # ComfyUI restarted under us
            if attempt > RETRIES or not _LOST.search(str(e)):
                raise
            time.sleep(RETRY_WAIT_S)
            continue
        if job.status in ("succeeded", "completed"):
            break
        err = str(job.error)
        if attempt > RETRIES or (job.error and not _LOST.search(err)):
            raise RuntimeError(f"{label or 'job'} {job.id} ended {job.status}: {err[:1500]}")
        time.sleep(RETRY_WAIT_S)
    saved: dict = {}
    for node_id, dest in outputs.items():
        dest = Path(dest)
        files = job.get_outputs(node_id)
        if not files:
            raise RuntimeError(f"{label or 'job'} {job.id}: node {node_id} produced no output")
        if dest.suffix:
            saved[node_id] = [files[0].to_file(dest)]
        else:
            dest.mkdir(parents=True, exist_ok=True)
            saved[node_id] = [f.to_file(dest / f.name) for f in files]
    return saved


# A shared GPU queue (e.g. a machine-wide GPU gate) hands the GPU out one ComfyUI prompt at a time, so several renders merged into one
# prompt get one turn instead of queueing once per render behind other projects' long jobs.
_NO_SHARE = {"LoadImage", "LoadVideo", "SaveImage", "SaveVideo", "SaveAudio", "PreviewImage"}


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def merge_graphs(graphs: list[dict]) -> tuple[dict, list[dict[str, str]]]:
    """Merge API graphs into one, sharing identical upstream nodes (model loaders, encoders).
    Returns the merged graph and, per input graph, a map old node id -> merged node id."""
    merged, by_sig, remaps = {}, {}, []
    for i, g in enumerate(graphs):
        remap, pending = {}, dict(g)
        while pending:
            ready = [n for n, node in pending.items()
                     if not any(_is_link(v) and v[0] in pending for v in node["inputs"].values())]
            if not ready:
                raise ValueError("graph has a cycle")
            for nid in ready:
                node = pending.pop(nid)
                ins = {k: ([remap[v[0]], v[1]] if _is_link(v) else v) for k, v in node["inputs"].items()}
                sig = json.dumps([node["class_type"], ins], sort_keys=True, default=str)
                if node["class_type"] not in _NO_SHARE and sig in by_sig:
                    remap[nid] = by_sig[sig]
                    continue
                new = f"b{i}_{nid}"
                merged[new] = {**node, "inputs": ins}
                remap[nid] = by_sig[sig] = new
        remaps.append(remap)
    return merged, remaps


def run_or_defer(defer: list | None, graph: dict, outputs: dict, **kw):
    """comfy.run, or queue the call on `defer` for run_many."""
    if defer is None:
        return run(graph, outputs, **kw)
    defer.append(dict(graph=graph, outputs=outputs, **kw))


def run_many(calls: list[dict], chunk: int = 3) -> None:
    """Run deferred comfy.run calls, `chunk` per ComfyUI prompt (calls to the same worker only)."""
    by_url: dict = {}
    for c in calls:
        by_url.setdefault(c.get("url"), []).append(c)
    for url, group in by_url.items():
        for s in range(0, len(group), max(1, chunk)):
            part = group[s:s + chunk]
            if len(part) == 1:
                run(**part[0])
                continue
            g, remaps = merge_graphs([c["graph"] for c in part])
            outputs = {remaps[j][n]: d for j, c in enumerate(part) for n, d in c["outputs"].items()}
            inputs = {(remaps[j][n], k): p for j, c in enumerate(part) for (n, k), p in (c.get("inputs") or {}).items()}
            label = " + ".join(c.get("label", "") for c in part)
            run(g, outputs, url=url, label=label, inputs=inputs, timeout=max(c.get("timeout", 7200) for c in part))
            first = Path(next(iter(outputs.values())))
            rec = first.with_name(first.stem + ".graph.json") if first.suffix else first / f"{label}.graph.json"
            for c in part[1:]:
                d = Path(next(iter(c["outputs"].values())))
                if d.suffix and rec.exists():
                    shutil.copy(rec, d.with_name(d.stem + ".graph.json"))


def chain(up: dict, up_out: str, down: dict, down_in: str) -> dict:
    """Feed node `up_out` (output 0) of graph `up` into graph `down` in place of its loader `down_in`.
    Node ids get a u_ / d_ prefix."""
    def pre(node, p, swap=None):
        ins = {k: ((swap if swap and v[0] == down_in else [p + v[0], v[1]]) if _is_link(v) else v)
               for k, v in node["inputs"].items()}
        return {**node, "inputs": ins}
    g = {"u_" + k: pre(v, "u_") for k, v in up.items()}
    g.update({"d_" + k: pre(v, "d_", ["u_" + up_out, 0]) for k, v in down.items() if k != down_in})
    return g
