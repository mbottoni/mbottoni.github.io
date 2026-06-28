"""
Animated GIF for scaling-ensembles.

Research question: do independently trained networks become more functionally
similar as width (parameter count) grows? This animation shows several
independently trained models as points in "function space" that start diverse
(scattered) and collapse toward a shared function as the network gets wider,
while a function-similarity gauge rises. Width ping-pongs so the GIF loops.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/scaling_ensembles_gif.py
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 380
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
ACCENT = (186, 57, 37)
BLUE = (33, 86, 165)
GREEN = (31, 150, 90)
TRACK = (228, 228, 224)

PALETTE = [(33, 86, 165), (186, 57, 37), (31, 150, 90),
           (150, 110, 60), (120, 80, 150), (40, 140, 160)]
K = 6
rng = np.random.default_rng(7)
DIRS = np.array([[math.cos(2 * math.pi * i / K), math.sin(2 * math.pi * i / K)] for i in range(K)])
JIT = rng.normal(0, 1, size=(K, 2))

PANEL = (56, 70, 410, 340)
WIDTHS = [8, 16, 32, 64, 128, 256, 512]


def load_font(size, bold=False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_CAP = load_font(15 * S)
F_SM = load_font(13 * S)
F_BIG = load_font(20 * S, bold=True)


def sc(p):
    return (p[0] * S, p[1] * S)


def text(dr, xy, s, font, fill, center=False):
    if center:
        bb = dr.textbbox((0, 0), s, font=font)
        xy = (xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))
    dr.text(sc(xy), s, font=font, fill=fill)


def frame(p):
    """p in [0,1]: 0 = narrow/diverse, 1 = wide/similar."""
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    text(dr, (W / 2, 22),
         "Do independently trained nets converge to the same function as they widen?",
         F_CAP, MUTED + (255,), center=True)

    # function-space panel
    dr.rounded_rectangle([sc(PANEL[:2]), sc(PANEL[2:])], radius=8 * S,
                         outline=(222, 222, 218, 255), width=S, fill=(250, 250, 248, 255))
    text(dr, ((PANEL[0] + PANEL[2]) / 2, PANEL[3] + 14), "function space", F_SM,
         MUTED + (255,), center=True)
    cx, cy = (PANEL[0] + PANEL[2]) / 2, (PANEL[1] + PANEL[3]) / 2
    spread = (1.0 - 0.86 * p) * 118

    pos = []
    for i in range(K):
        x = cx + spread * DIRS[i][0] + 0.18 * spread * JIT[i][0]
        y = cy + spread * DIRS[i][1] + 0.18 * spread * JIT[i][1]
        pos.append((x, y))
    # faint links to the shared centroid
    for i in range(K):
        dr.line([sc(pos[i]), sc((cx, cy))], fill=(205, 205, 200, 160), width=S)
    # shared-function marker
    dr.ellipse([sc((cx - 4, cy - 4)), sc((cx + 4, cy + 4))], fill=(150, 150, 145, 255))
    # model points
    for i in range(K):
        x, y = pos[i]
        dr.ellipse([sc((x - 5.5, y - 5.5)), sc((x + 5.5, y + 5.5))],
                   fill=PALETTE[i] + (255,), outline=BG + (255,), width=S)
    text(dr, (PANEL[0] + 10, PANEL[1] + 8), "6 independently trained models", F_SM,
         MUTED + (255,))

    # right column: width readout + similarity gauge
    rx = 470
    w_idx = int(round(p * (len(WIDTHS) - 1)))
    text(dr, (rx, 96), "network width", F_SM, MUTED + (255,))
    text(dr, (rx, 112), str(WIDTHS[w_idx]), F_BIG, BLUE + (255,))
    # width bar
    bx0, bx1, by = rx, rx + 180, 150
    dr.rounded_rectangle([sc((bx0, by - 7)), sc((bx1, by + 7))], radius=7 * S, fill=TRACK + (255,))
    dr.rounded_rectangle([sc((bx0, by - 7)), sc((bx0 + (bx1 - bx0) * p, by + 7))],
                         radius=7 * S, fill=BLUE + (255,))

    sim = 0.18 + 0.78 * p
    text(dr, (rx, 196), "function similarity", F_SM, MUTED + (255,))
    gy = 226
    dr.rounded_rectangle([sc((bx0, gy - 9)), sc((bx1, gy + 9))], radius=9 * S, fill=TRACK + (255,))
    dr.rounded_rectangle([sc((bx0, gy - 9)), sc((bx0 + (bx1 - bx0) * sim, gy + 9))],
                         radius=9 * S, fill=GREEN + (255,))
    text(dr, (bx1 + 12, gy - 9), f"{sim:.0%}", F_SM, GREEN + (255,))

    text(dr, (rx, 268),
         "diversity collapses" if p > 0.6 else "diverse minima",
         F_SM, (ACCENT if p > 0.6 else MUTED) + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    n = 34
    fwd = [i / (n - 1) for i in range(n)]
    seq = fwd + [1.0] * 6 + fwd[::-1] + [0.0] * 4
    frames = [frame(p) for p in seq]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "scaling-ensembles.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
