"""
Animated GIF for the flow-match project (Flow Matching for generative modeling).

Shows the canonical flow-matching picture: a cloud of Gaussian *noise* is
transported into a *two-moons* data distribution along the conditional flow path
x_t = (1-t)·x0 + t·x1, the same thing `FlowMatcher.sample_trajectory` produces.
t ping-pongs 0->1->0 so the GIF loops seamlessly (and hints at the flow's
reversibility).

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/flow_match_gif.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 620, 380
S = 2
BG = (253, 253, 252)
MUTED = (120, 128, 140)
BLUE = (33, 86, 165)
RED = (186, 57, 37)

N = 460
rng = np.random.default_rng(0)


def two_moons(n):
    """sklearn-style two moons, returned with a 0/1 label per point."""
    half = n // 2
    t0 = np.linspace(0, np.pi, half)
    t1 = np.linspace(0, np.pi, n - half)
    outer = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    inner = np.stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5], axis=1)
    xy = np.concatenate([outer, inner], axis=0)
    labels = np.array([0] * half + [1] * (n - half))
    xy = xy + rng.normal(0, 0.045, size=xy.shape)
    return xy, labels


def load_font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/HelveticaNeue.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_CAP = load_font(15 * S)
F_SM = load_font(13 * S)

TARGET, LABELS = two_moons(N)
NOISE = rng.normal(0, 0.6, size=(N, 2)) + np.array([0.5, 0.0])

# fixed affine mapping data-space -> plot box (fit to both endpoints)
_all = np.concatenate([TARGET, NOISE], axis=0)
_lo, _hi = _all.min(0), _all.max(0)
PLOT = (60, 58, W - 60, H - 70)
PAD = 16


def to_canvas(pts):
    fx = (pts[:, 0] - _lo[0]) / (_hi[0] - _lo[0] + 1e-9)
    fy = (pts[:, 1] - _lo[1]) / (_hi[1] - _lo[1] + 1e-9)
    x = PLOT[0] + PAD + fx * (PLOT[2] - PLOT[0] - 2 * PAD)
    y = PLOT[3] - PAD - fy * (PLOT[3] - PLOT[1] - 2 * PAD)   # flip y
    return np.stack([x, y], axis=1)


def sc(p):
    return (p[0] * S, p[1] * S)


def text_c(dr, xy, s, font, fill):
    bb = dr.textbbox((0, 0), s, font=font)
    dr.text(sc((xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))),
            s, font=font, fill=fill)


def frame(t):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    text_c(dr, (W / 2, 28), "Flow matching: transporting noise to data", F_CAP, MUTED + (255,))

    pts = to_canvas((1 - t) * NOISE + t * TARGET)
    # colour fades from neutral (noise) to per-moon colour (data)
    for i in range(N):
        base = np.array(BLUE if LABELS[i] == 0 else RED, dtype=float)
        col = tuple(int(c) for c in (160 + (base - 160) * t))
        x, y = pts[i]
        dr.ellipse([sc((x - 2.6, y - 2.6)), sc((x + 2.6, y + 2.6))], fill=col + (210,))

    # time bar: noise -- t -- data
    by = H - 38
    bx0, bx1 = 180, 440
    dr.line([sc((bx0, by)), sc((bx1, by))], fill=(210, 210, 205, 255), width=2 * S)
    mx = bx0 + (bx1 - bx0) * t
    dr.ellipse([sc((mx - 5, by - 5)), sc((mx + 5, by + 5))], fill=MUTED + (255,))
    text_c(dr, (bx0 - 28, by), "noise", F_SM, MUTED + (255,))
    text_c(dr, (bx1 + 26, by), "data", F_SM, MUTED + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    def ease(u):
        return u * u * (3 - 2 * u)        # smoothstep

    fwd = [ease(i / 33) for i in range(34)]
    seq = fwd + [1.0] * 6 + fwd[::-1] + [0.0] * 4
    frames = [frame(t) for t in seq]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "flow-match.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
