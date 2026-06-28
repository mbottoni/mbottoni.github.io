"""
Animated GIF for kalman-transformer ("Kalman <-> Transformer for Portfolio
Optimization").

The project frames expected-return estimation as state estimation: noisy returns
are observations of a hidden expected return, recovered by an interchangeable
estimator -- a local-level Kalman filter or a learned Transformer -- which then
feeds a Markowitz optimizer. This animation shows that core picture: a scrolling
stream of noisy returns with two denoised expected-return estimates (Kalman and
Transformer) tracking the hidden trend.

The signal is periodic, so the scroll loops seamlessly.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/kalman_transformer_gif.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 340
S = 2
BG = (253, 253, 252)
MUTED = (120, 128, 140)
GRID = (232, 232, 228)
DOT = (165, 172, 182)
BLUE = (33, 86, 165)        # Kalman
RED = (186, 57, 37)         # Transformer

F = 60                       # period (frames)
WPTS = 78                    # visible points

rng = np.random.default_rng(4)
_i = np.arange(F)
MU = (0.9 * np.sin(2 * np.pi * _i / F)
      + 0.45 * np.sin(2 * np.pi * 2 * _i / F + 1.0)
      + 0.22 * np.sin(2 * np.pi * 3 * _i / F + 2.0))
NOISE = rng.normal(0, 0.4, size=F)
OBS = MU + NOISE
KAL = 0.70 * MU + 0.30 * np.roll(MU, 1)             # smoothed (lags slightly)
TRF = 0.66 * MU + 0.34 * np.roll(MU, -1) + 0.04     # learned (leads slightly)

YMIN, YMAX = OBS.min() - 0.2, OBS.max() + 0.2
PLOT = (54, 70, W - 54, H - 46)


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


def sc(p):
    return (p[0] * S, p[1] * S)


def px(k):
    return PLOT[0] + (PLOT[2] - PLOT[0]) * k / (WPTS - 1)


def py(v):
    fy = (v - YMIN) / (YMAX - YMIN)
    return PLOT[3] - fy * (PLOT[3] - PLOT[1])


def text(dr, xy, s, font, fill, center=False):
    if center:
        bb = dr.textbbox((0, 0), s, font=font)
        xy = (xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))
    dr.text(sc(xy), s, font=font, fill=fill)


def polyline(dr, vals, color, width=2):
    pts = [sc((px(k), py(vals[k]))) for k in range(WPTS)]
    dr.line(pts, fill=color + (255,), width=width * S, joint="curve")


def frame(f):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")

    text(dr, (W / 2, 22), "Estimating expected returns: a Kalman filter and a Transformer",
         F_CAP, MUTED + (255,), center=True)

    # zero line
    dr.line([sc((PLOT[0], py(0))), sc((PLOT[2], py(0)))], fill=GRID + (255,), width=S)

    idx = (f + np.arange(WPTS)) % F
    obs, kal, trf = OBS[idx], KAL[idx], TRF[idx]

    # noisy observations (returns)
    for k in range(WPTS):
        x, y = px(k), py(obs[k])
        dr.ellipse([sc((x - 2.4, y - 2.4)), sc((x + 2.4, y + 2.4))], fill=DOT + (220,))

    polyline(dr, kal, BLUE, 3)
    polyline(dr, trf, RED, 3)

    # legend
    lx, ly = PLOT[0] + 6, PLOT[1] + 4
    items = [(DOT, "noisy returns"), (BLUE, "Kalman"), (RED, "Transformer")]
    for i, (col, label) in enumerate(items):
        yy = ly + i * 20
        dr.rounded_rectangle([sc((lx, yy)), sc((lx + 16, yy + 9))], radius=2 * S,
                             fill=col + (255,))
        text(dr, (lx + 24, yy - 3), label, F_SM, MUTED + (255,))

    text(dr, (W / 2, H - 30),
         "→ denoised expected return feeds the Markowitz optimizer",
         F_SM, MUTED + (255,), center=True)

    return img.resize((W, H), Image.LANCZOS)


def main():
    frames = [frame(f) for f in range(F)]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "kalman-transformer.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
