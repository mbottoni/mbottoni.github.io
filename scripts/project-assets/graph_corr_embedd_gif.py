"""
Animated GIF for graph-corr-embedd ("Measuring Graph Correlation from Graph
Embeddings").

Depicts the core idea: a correlated pair of graphs (B is a rewired copy of A) is
mapped to embedding point-clouds; as the correlation rho drops, B's non-shared
edges fade, the two embedding clouds drift apart, and a gauge tracks the measured
graph correlation. rho sweeps high -> low -> high so the GIF loops.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/graph_corr_embedd_gif.py
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 780, 320
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
BLUE = (33, 86, 165)       # graph / embedding A
RED = (186, 57, 37)        # graph / embedding B
EDGE = (150, 158, 170)
TRACK = (228, 228, 224)

rng = random.Random(5)
NODES = [(rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)) for _ in range(9)]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (2, 5),
         (5, 6), (6, 7), (4, 7), (7, 8), (1, 8), (0, 4)]
# each edge "survives" in B while rho exceeds its threshold
_th = [0.30 + 0.62 * i / (len(EDGES) - 1) for i in range(len(EDGES))]
rng.shuffle(_th)
EDGE_THRESH = _th

_nr = np.random.default_rng(2)
CLOUD = _nr.normal(0, 1, size=(16, 2))   # shared offsets for both clouds


def load_font(size, bold=False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F = load_font(15 * S)
F_BOLD = load_font(15 * S, bold=True)
F_SM = load_font(13 * S)


def sc(p):
    return (p[0] * S, p[1] * S)


def text_c(dr, xy, s, font, fill):
    bb = dr.textbbox((0, 0), s, font=font)
    dr.text(sc((xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))),
            s, font=font, fill=fill)


def node_xy(i, box):
    x0, y0, x1, y1 = box
    nx, ny = NODES[i]
    return (x0 + nx * (x1 - x0), y0 + ny * (y1 - y0))


def draw_graph(dr, box, rho, color, is_b):
    for e, (a, b) in enumerate(EDGES):
        pa, pb = node_xy(a, box), node_xy(b, box)
        if is_b and rho < EDGE_THRESH[e]:
            col = (210, 210, 205, 255)        # faded: edge no longer shared
        else:
            col = EDGE + (255,)
        dr.line([sc(pa), sc(pb)], fill=col, width=2 * S)
    for i in range(len(NODES)):
        x, y = node_xy(i, box)
        r = 4
        dr.ellipse([sc((x - r, y - r)), sc((x + r, y + r))], fill=color + (255,),
                   outline=BG + (255,), width=S)


def draw_cloud(dr, center, color):
    for dx, dy in CLOUD:
        x, y = center[0] + dx * 16, center[1] + dy * 16
        dr.ellipse([sc((x - 3.2, y - 3.2)), sc((x + 3.2, y + 3.2))], fill=color + (200,))


def arrow(dr, p0, p1, color, width=2):
    dr.line([sc(p0), sc(p1)], fill=color, width=width * S)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for off in (math.radians(150), -math.radians(150)):
        x = p1[0] + 8 * math.cos(ang + off)
        y = p1[1] + 8 * math.sin(ang + off)
        dr.line([sc((x, y)), sc(p1)], fill=color, width=width * S)


def frame(rho):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")

    text_c(dr, (W / 2, 24), "Measuring graph correlation from graph embeddings",
           F, MUTED + (255,))

    # two graphs (A on top, B below)
    box_a = (60, 56, 250, 150)
    box_b = (60, 168, 250, 262)
    text_c(dr, (35, 103), "A", F_BOLD, BLUE + (255,))
    text_c(dr, (35, 215), "B", F_BOLD, RED + (255,))
    draw_graph(dr, box_a, rho, BLUE, is_b=False)
    draw_graph(dr, box_b, rho, RED, is_b=True)

    # embed arrow
    arrow(dr, (262, 159), (322, 159), MUTED + (255,))
    text_c(dr, (292, 145), "embed", F_SM, MUTED + (255,))

    # latent space box + clouds
    lx0, ly0, lx1, ly1 = 340, 56, 700, 262
    dr.rounded_rectangle([sc((lx0, ly0)), sc((lx1, ly1))], radius=10 * S,
                         outline=(220, 220, 215, 255), width=S, fill=(250, 250, 248, 255))
    text_c(dr, ((lx0 + lx1) / 2, ly0 + 14), "embedding space", F_SM, MUTED + (255,))
    cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2 + 8
    sep = (1 - rho)
    # correlated (rho->1): clouds coincide; uncorrelated: they pull apart
    a_center = (cx - 70 * sep, cy - 6 * sep)
    b_center = (cx + 70 * sep, cy + 18 * sep)
    draw_cloud(dr, a_center, BLUE)
    draw_cloud(dr, b_center, RED)

    # correlation gauge
    gy = 296
    gx0, gx1 = 300, 560
    dr.rounded_rectangle([sc((gx0, gy - 9)), sc((gx1, gy + 9))], radius=9 * S, fill=TRACK + (255,))
    fillw = (gx1 - gx0) * max(rho, 0.02)
    dr.rounded_rectangle([sc((gx0, gy - 9)), sc((gx0 + fillw, gy + 9))], radius=9 * S,
                         fill=BLUE + (255,))
    text_c(dr, (210, gy), "graph correlation", F_SM, MUTED + (255,))
    dr.text(sc((gx1 + 14, gy - 9)), f"ρ = {rho:.2f}", font=F, fill=BLUE + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    # rho sweeps high -> low -> high (smooth loop)
    n = 30
    seq = ([0.95 - 0.8 * (i / (n - 1)) for i in range(n)] +
           [0.15 + 0.8 * (i / (n - 1)) for i in range(n)])
    frames = [frame(r) for r in seq]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "graph-corr-embedd.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
