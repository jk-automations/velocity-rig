#!/usr/bin/env python
"""Validate API-format workflows offline against a ComfyUI /object_info snapshot. Nothing is queued, no server is needed.

  ~/ComfyUI/venv/bin/python scripts/validate_workflows.py --object-info ~/tmp/oi_8188.json --pack custom_nodes/<pack> workflows/*_api.json

Checks per node: class known, required inputs present, links point at an existing node and output slot, combo values (model
file names, samplers) exist in the snapshot, at least one output node. The pack's own classes are not in a stock snapshot, so
their specs are read from the pack's INPUT_TYPES / RETURN_TYPES (import needs torch: use ComfyUI's venv python).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

UPLOADED = {("LoadImage", "image"), ("LoadVideo", "file"), ("LoadAudio", "audio")}


def pack_specs(pack: Path) -> dict:
    spec = importlib.util.spec_from_file_location("rig_pack", pack / "__init__.py", submodule_search_locations=[str(pack)])
    m = importlib.util.module_from_spec(spec)
    sys.modules["rig_pack"] = m
    spec.loader.exec_module(m)
    out = {}
    for name, cls in m.NODE_CLASS_MAPPINGS.items():
        it = cls.INPUT_TYPES()
        norm = lambda d: {k: [list(v[0]) if isinstance(v[0], (list, tuple)) else v[0], v[1] if len(v) > 1 else {}] for k, v in (d or {}).items()}
        out[name] = {"input": {"required": norm(it.get("required")), "optional": norm(it.get("optional"))},
                     "output": list(cls.RETURN_TYPES), "output_node": bool(getattr(cls, "OUTPUT_NODE", False))}
    return out


def check(path: Path, oi: dict) -> list[str]:
    g = json.loads(path.read_text())
    g = g.get("graph", g) if isinstance(g.get("graph"), dict) else g  # *.graph.json records wrap the API graph
    errs = []
    if not any(oi.get(n["class_type"], {}).get("output_node") for n in g.values()):
        errs.append("no output node")
    for nid, n in g.items():
        ct = n["class_type"]
        spec = oi.get(ct)
        if not spec:
            errs.append(f"{nid} {ct}: unknown class")
            continue
        ins = n.get("inputs", {})
        for name, d in spec["input"].get("required", {}).items():
            if name not in ins:
                errs.append(f"{nid} {ct}: missing required input {name}")
        for name, v in ins.items():
            d = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}.get(name)
            if d is None:
                errs.append(f"{nid} {ct}: unexpected input {name}")
            elif isinstance(v, list):
                src = g.get(str(v[0]))
                if not src:
                    errs.append(f"{nid} {ct}.{name}: link to missing node {v[0]}")
                elif v[1] >= len(oi.get(src["class_type"], {}).get("output", [None] * 99)):
                    errs.append(f"{nid} {ct}.{name}: node {v[0]} has no output {v[1]}")
            elif isinstance(d[0], list) and (ct, name) not in UPLOADED and v not in d[0]:
                errs.append(f"{nid} {ct}.{name}: {v!r} not in {d[0][:6]}...")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-info", required=True)
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    oi = json.loads(Path(a.object_info).expanduser().read_text())
    for p in a.pack:
        oi.update(pack_specs(Path(p)))
    bad = 0
    for f in a.files:
        e = check(Path(f), oi)
        print(f"{f}: {'VALID' if not e else 'INVALID'}")
        for x in e[:12]:
            print("  ", x)
        bad += bool(e)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
