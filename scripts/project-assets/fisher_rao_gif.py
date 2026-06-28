"""
Animated GIF for fisher-rao-ml (Fisher-Rao geodesic distance vs KL divergence).

On the manifold of Bernoulli distributions, a distribution p maps to the unit
vector (sqrt(p), sqrt(1-p)) -- a point on the unit quarter-circle. The Fisher-Rao
distance between two distributions is then the geodesic *arc* between their
points (symmetric), while KL is a flat, asymmetric divergence
(KL(A||B) != KL(B||A)). The animation sweeps B around the manifold and contrasts
the two. B ping-pongs so the GIF loops.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/fisher_rao_gif.py
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
ACCENT = (186, 57, 37)        # Fisher-Rao
BLUE = (33, 86, 165)          # distribution A / KL
TRACK = (228, 228, 224)
CHORD = (200, 200, 196)

OX, OY, R = 96, 312, 210      # manifold origin + radius
A_ANG = math.radians(28)      # fixed distribution A


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
F_LBL = load_font(12 * S)


def sc(p):
    return (p[0] * S, p[1] * S)


def on_arc(ang):
    return (OX + R * math.cos(ang), OY - R * math.sin(ang))


def text(dr, xy, s, font, fill, center=False):
    if center:
        bb = dr.textbbox((0, 0), s, font=font)
        xy = (xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))
    dr.text(sc(xy), s, font=font, fill=fill)


def kl(p, q):
    p = min(max(p, 1e-3), 1 - 1e-3)
    q = min(max(q, 1e-3), 1 - 1e-3)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def gauge(dr, x, y, frac, color, label, value):
    w = 168
    dr.rounded_rectangle([sc((x, y - 8)), sc((x + w, y + 8))], radius=8 * S, fill=TRACK + (255,))
    dr.rounded_rectangle([sc((x, y - 8)), sc((x + w * min(frac, 1), y + 8))], radius=8 * S,
                         fill=color + (255,))
    text(dr, (x, y - 26), label, F_LBL, MUTED + (255,))
    text(dr, (x + w + 10, y - 8), value, F_SM, color + (255,))


def mini_dist(dr, x, y, p, color, name):
    """two-bar Bernoulli distribution glyph."""
    bw, bh = 18, 44
    for i, val in enumerate((p, 1 - p)):
        bx = x + i * (bw + 6)
        dr.rectangle([sc((bx, y + bh - val * bh)), sc((bx + bw, y + bh))], fill=color + (255,))
        dr.rectangle([sc((bx, y)), sc((bx + bw, y + bh))], outline=(210, 210, 206, 255), width=S)
    text(dr, (x, y + bh + 6), name, F_LBL, MUTED + (255,))


def frame(b_ang):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    text(dr, (W / 2, 22), "Fisher-Rao geodesic vs KL divergence on the space of distributions",
         F_CAP, MUTED + (255,), center=True)

    # axes + manifold arc (unit quarter circle of Bernoulli distributions)
    dr.line([sc((OX, OY)), sc((OX + R + 16, OY))], fill=(215, 215, 211, 255), width=S)
    dr.line([sc((OX, OY)), sc((OX, OY - R - 16))], fill=(215, 215, 211, 255), width=S)
    text(dr, (OX + R - 8, OY + 8), "√p", F_LBL, MUTED + (255,))
    text(dr, (OX - 30, OY - R - 4), "√(1−p)", F_LBL, MUTED + (255,))
    box = [sc((OX - R, OY - R)), sc((OX + R, OY + R))]
    dr.arc(box, start=-90, end=0, fill=(180, 185, 195, 255), width=2 * S)

    A = on_arc(A_ANG)
    B = on_arc(b_ang)

    # Euclidean chord (the "flat" view) — faint
    dr.line([sc(A), sc(B)], fill=CHORD + (255,), width=2 * S)
    # Fisher-Rao geodesic = arc between A and B
    lo, hi = sorted((A_ANG, b_ang))
    dr.arc(box, start=-math.degrees(hi), end=-math.degrees(lo), fill=ACCENT + (255,), width=4 * S)

    for pt, col, name in ((A, BLUE, "A"), (B, ACCENT, "B")):
        dr.ellipse([sc((pt[0] - 6, pt[1] - 6)), sc((pt[0] + 6, pt[1] + 6))],
                   fill=col + (255,), outline=BG + (255,), width=S)
        text(dr, (pt[0] + 8, pt[1] - 18), name, F_SM, col + (255,))

    # right column
    pA = math.cos(A_ANG) ** 2
    pB = math.cos(b_ang) ** 2
    fr = abs(b_ang - A_ANG)                      # geodesic angle (radians)
    kab, kba = kl(pA, pB), kl(pB, pA)
    kmax = 2.5

    rx = 430
    mini_dist(dr, rx, 70, pA, BLUE, "A")
    mini_dist(dr, rx + 90, 70, pB, ACCENT, "B")

    gauge(dr, rx, 190, fr / (math.pi / 2), ACCENT,
          "Fisher-Rao distance  (symmetric)", f"{fr:.2f}")
    gauge(dr, rx, 244, kab / kmax, BLUE, "KL(A ‖ B)", f"{kab:.2f}")
    gauge(dr, rx, 290, kba / kmax, (90, 140, 200), "KL(B ‖ A)", f"{kba:.2f}")
    text(dr, (rx, 318), "KL is asymmetric — the two values differ", F_LBL, MUTED + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    lo, hi = math.radians(8), math.radians(82)
    n = 36
    fwd = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    seq = fwd + [hi] * 4 + fwd[::-1] + [lo] * 4
    frames = [frame(b) for b in seq]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "fisher-rao-ml.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
