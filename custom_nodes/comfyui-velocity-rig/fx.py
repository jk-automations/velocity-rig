"""Velocity FX: in-camera speed effects for a decoded clip (IMAGE batch, T x H x W x C, 0..1).

shake       rig vibration: smooth multi-frequency jitter + a little roll, with overscan so edges never show
speed_blur  radial zoom blur that leaves the center sharp and smears the edges, plus a touch of frame smear
zoom_punch  crash zoom: snaps in over the first quarter of the shot, then holds
chroma      lens chromatic fringing toward the edges
ramp        speed ramp with the same frame count: fast in, slow at the middle of the shot, fast out
All amounts are 0..1. The frame count never changes, so LTX's audio stays in sync.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

KEYS = ("shake", "speed_blur", "zoom_punch", "chroma", "ramp")


def _warp_time(x: torch.Tensor, ramp: float) -> torch.Tensor:
    t = x.shape[0]
    if ramp <= 0 or t < 3:
        return x
    a = 0.6 * min(ramp, 1.0)
    u = torch.linspace(0, 1, t)
    src = (u + a * torch.sin(2 * math.pi * u) / (2 * math.pi)).clamp(0, 1) * (t - 1)
    lo = src.floor().long().clamp(0, t - 1)
    hi = (lo + 1).clamp(max=t - 1)
    w = (src - lo.float()).view(-1, 1, 1, 1)
    return x[lo] * (1 - w) + x[hi] * w


def _sample(x: torch.Tensor, scale: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """x: N,C,H,W. scale > 1 zooms in. tx, ty in normalized units, rot in radians. All length N."""
    inv = 1.0 / scale
    cos, sin = torch.cos(rot) * inv, torch.sin(rot) * inv
    theta = torch.stack([torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def apply(images: torch.Tensor, shake=0.0, speed_blur=0.0, zoom_punch=0.0, chroma=0.0, ramp=0.0, seed=0,
          fps=24.0, chunk=8) -> torch.Tensor:
    if images.ndim != 4 or images.shape[0] == 0:
        return images
    T, H, W, _ = images.shape
    x = _warp_time(images.float(), ramp)
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    t = torch.arange(T, dtype=torch.float32) / float(fps)

    # camera path: overscan + crash zoom + jitter
    scale = torch.full((T,), 1.0 + 0.04 * shake)
    if zoom_punch > 0:
        u = (torch.arange(T, dtype=torch.float32) / max(T - 1, 1) / 0.25).clamp(0, 1)
        scale = scale * (1.0 + 0.45 * zoom_punch * (u * u * (3 - 2 * u)))
    tx = torch.zeros(T); ty = torch.zeros(T); rot = torch.zeros(T)
    if shake > 0:
        for hz, amp in ((2.3, 1.0), (5.1, 0.6), (11.7, 0.35), (17.9, 0.2)):
            ph = torch.rand(3, generator=g) * 2 * math.pi
            tx += amp * torch.sin(2 * math.pi * hz * t + ph[0])
            ty += amp * torch.sin(2 * math.pi * hz * 1.13 * t + ph[1])
            rot += amp * torch.sin(2 * math.pi * hz * 0.87 * t + ph[2])
        tx *= 0.012 * shake; ty *= 0.016 * shake; rot *= math.radians(0.5) * shake

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    r = torch.sqrt((xx * W / H) ** 2 + yy ** 2) / math.sqrt((W / H) ** 2 + 1)
    edge = (r / 0.75).clamp(0, 1) ** 1.6
    vignette = 1.0 - 0.18 * (r.clamp(0, 1) ** 2) * min(1.0, 0.4 + speed_blur)

    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        c = x[s:e].permute(0, 3, 1, 2)
        n = e - s
        sc, a, b, ro = scale[s:e], tx[s:e], ty[s:e], rot[s:e]
        base = _sample(c, sc, a, b, ro)
        if speed_blur > 0:
            taps = 6
            acc = torch.zeros_like(base)
            for k in range(taps):
                acc += _sample(c, sc * (1.0 + 0.018 * speed_blur * k), a, b, ro)
            base = base + (acc / taps - base) * (edge * min(1.0, speed_blur * 1.2))
        if chroma > 0:
            d = 0.003 * chroma
            rch = _sample(c[:, :1], sc * (1 + d), a, b, ro)
            bch = _sample(c[:, 2:3], sc * (1 - d), a, b, ro)
            base = torch.cat([base[:, :1] + (rch - base[:, :1]) * edge, base[:, 1:2],
                              base[:, 2:3] + (bch - base[:, 2:3]) * edge], 1)
        base = base * vignette
        out.append(base.permute(0, 2, 3, 1))
    y = torch.cat(out, 0)
    if speed_blur > 0 and T > 1:  # light frame smear, reads as shutter drag at speed
        k = 0.22 * speed_blur
        y = torch.cat([y[:1], y[1:] * (1 - k) + y[:-1] * k], 0)
    return y.clamp(0, 1).to(images.dtype)
