"""
Animated GIF for foundational-brain (a self-supervised foundation model for fMRI).

The project's whole picture in one loop: an fMRI frame (ROI activity) is
compressed by the encoder into a latent z_t, a recurrent core steps the latent
forward to a predicted z_{t+1}, and the decoder maps that back to ROI space --
where the one-step-ahead forecast is compared against the truth.

Every signal is built on a period-F grid and indexed mod F, so the scroll and
the latent trajectory both loop seamlessly.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/foundational_brain_gif.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 380
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
FAINT = (196, 202, 210)
BORDER = (224, 226, 222)
BLUE = (33, 86, 165)          # truth / observed latent
RED = (186, 57, 37)           # model prediction
COLD = (28, 74, 142)          # raster negative
WARM = (176, 48, 32)          # raster positive

F = 48                        # period (frames)
NROI = 16                     # rows in the raster
WCOLS = 40                    # visible columns of the raster
WPTS = 44                     # visible points of the forecast trace
LEVELS = 13                   # quantization steps of the raster colormap

PT, PB = 78, 252              # panel top / bottom
P1 = (34, 248)                # raster panel x-range
P2 = (292, 470)               # latent panel x-range
P3 = (514, 690)               # forecast panel x-range

# --- data -------------------------------------------------------------------
# Three slow latent factors drive every ROI; the model's job is their dynamics.
rng = np.random.default_rng(7)
_th = 2 * np.pi * np.arange(F) / F
Z1 = np.sin(_th) + 0.16 * np.sin(3 * _th + 0.4)
Z2 = np.sin(2 * _th + 0.9)
Z3 = np.cos(_th + 1.3) + 0.25 * np.cos(3 * _th)

LOAD = rng.normal(0, 1, size=(NROI, 3))
NOISE = rng.normal(0, 0.22, size=(NROI, F))
X = LOAD @ np.vstack([Z1, Z2, Z3]) + NOISE
X /= np.abs(X).max()

# The forecast trace: one ROI, truth vs the model's one-step-ahead guess. Lightly
# smoothed so the shape reads at this size; the prediction tracks it closely but
# lags a touch, as a real RNN does.
R0 = 4
_k = np.array([0.25, 0.5, 0.25])
TRUE = sum(w * np.roll(X[R0], s) for w, s in zip(_k, (-1, 0, 1)))
PRED = 0.88 * np.roll(TRUE, 1) + 0.12 * np.roll(TRUE, 2) + 0.02
TMIN, TMAX = min(TRUE.min(), PRED.min()) - 0.12, max(TRUE.max(), PRED.max()) + 0.12

LMIN1, LMAX1 = Z1.min() - 0.35, Z1.max() + 0.35
LMIN2, LMAX2 = Z2.min() - 0.35, Z2.max() + 0.35


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
F_SM = load_font(12 * S)
F_TINY = load_font(11 * S)
F_BOLD = load_font(13 * S, bold=True)


def sc(p):
    return (p[0] * S, p[1] * S)


def text(dr, xy, s, font, fill, center=False):
    if center:
        bb = dr.textbbox((0, 0), s, font=font)
        xy = (xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))
    dr.text(sc(xy), s, font=font, fill=fill)


def diverging(v):
    """Map v in [-1, 1] to a cold-white-warm colormap (neuroimaging convention).

    Quantized to LEVELS steps: it keeps the GIF palette small, and the banding
    reads as a deliberate discrete scale rather than a gradient.
    """
    v = float(np.clip(v, -1, 1))
    step = round(v * (LEVELS // 2)) / (LEVELS // 2)
    base = COLD if step < 0 else WARM
    k = abs(step)
    return tuple(int(255 + (base[i] - 255) * k) for i in range(3))


def arrow(dr, x0, x1, y, label):
    dr.line([sc((x0, y)), sc((x1 - 7, y))], fill=FAINT + (255,), width=2 * S)
    dr.polygon([sc((x1, y)), sc((x1 - 8, y - 4.5)), sc((x1 - 8, y + 4.5))],
               fill=FAINT + (255,))
    text(dr, ((x0 + x1) / 2, y - 20), label, F_SM, MUTED + (255,), center=True)


def panel_frame(dr, xr):
    dr.rectangle([sc((xr[0], PT)), sc((xr[1], PB))], outline=BORDER + (255,), width=S)


def draw_raster(dr, f):
    x0, x1 = P1
    cw = (x1 - x0) / WCOLS
    ch = (PB - PT) / NROI
    idx = (f + np.arange(WCOLS)) % F
    for c in range(WCOLS):
        col = X[:, idx[c]]
        for r in range(NROI):
            dr.rectangle(
                [sc((x0 + c * cw, PT + r * ch)),
                 sc((x0 + (c + 1) * cw + 0.6, PT + (r + 1) * ch + 0.6))],
                fill=diverging(col[r]) + (255,),
            )
    # the frame currently being encoded
    dr.rectangle([sc((x1 - cw, PT)), sc((x1, PB))], outline=INK + (255,), width=S)
    text(dr, (x1 - cw / 2, PT - 15), "x_t", F_TINY, INK + (255,), center=True)
    panel_frame(dr, P1)


def draw_latent(dr, f):
    x0, x1 = P2

    def lx(v):
        return x0 + 16 + (x1 - x0 - 32) * (v - LMIN1) / (LMAX1 - LMIN1)

    def ly(v):
        return PB - 16 - (PB - PT - 32) * (v - LMIN2) / (LMAX2 - LMIN2)

    panel_frame(dr, P2)
    loop = [sc((lx(Z1[k]), ly(Z2[k]))) for k in range(F)]
    dr.line(loop + [loop[0]], fill=FAINT + (255,), width=2 * S, joint="curve")

    now = (f + WCOLS - 1) % F
    nxt = (now + 1) % F
    ax, ay = lx(Z1[now]), ly(Z2[now])
    bx, by = lx(Z1[nxt]), ly(Z2[nxt])

    # a short fading trail behind the current state
    for j in range(1, 9):
        k = (now - j) % F
        a = int(150 * (1 - j / 9))
        px, py = lx(Z1[k]), ly(Z2[k])
        dr.ellipse([sc((px - 3, py - 3)), sc((px + 3, py + 3))], fill=BLUE + (a,))

    dr.line([sc((ax, ay)), sc((bx, by))], fill=RED + (255,), width=3 * S)
    dr.ellipse([sc((bx - 5, by - 5)), sc((bx + 5, by + 5))], fill=RED + (255,))
    dr.ellipse([sc((ax - 5.5, ay - 5.5)), sc((ax + 5.5, ay + 5.5))], fill=BLUE + (255,))

    text(dr, (x0 + 10, PT + 8), "z_t", F_TINY, BLUE + (255,))
    text(dr, (x1 - 74, PT + 8), "RNN step -> z_t+1", F_TINY, RED + (255,))


def draw_forecast(dr, f):
    x0, x1 = P3

    def fx(k):
        return x0 + 12 + (x1 - x0 - 24) * k / (WPTS - 1)

    def fy(v):
        return PB - 14 - (PB - PT - 28) * (v - TMIN) / (TMAX - TMIN)

    panel_frame(dr, P3)
    idx = (f + np.arange(WPTS)) % F
    tru = [sc((fx(k), fy(TRUE[idx[k]]))) for k in range(WPTS)]
    prd = [sc((fx(k), fy(PRED[idx[k]]))) for k in range(WPTS)]
    dr.line(tru, fill=BLUE + (255,), width=3 * S, joint="curve")
    dr.line(prd, fill=RED + (255,), width=3 * S, joint="curve")

    ly = PT + 8
    for i, (col, label) in enumerate([(BLUE, "truth"), (RED, "predicted")]):
        yy = ly + i * 16
        dr.rounded_rectangle([sc((x0 + 12, yy)), sc((x0 + 26, yy + 8))],
                             radius=2 * S, fill=col + (255,))
        text(dr, (x0 + 32, yy - 3), label, F_TINY, MUTED + (255,))


def frame(f):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")

    text(dr, (W / 2, 20),
         "A self-supervised foundation model for fMRI dynamics",
         F_CAP, INK + (255,), center=True)
    text(dr, (W / 2, 42),
         "compress each frame, step the latent forward, decode the next moment",
         F_SM, MUTED + (255,), center=True)

    draw_raster(dr, f)
    draw_latent(dr, f)
    draw_forecast(dr, f)

    mid = (PT + PB) / 2
    arrow(dr, P1[1] + 8, P2[0] - 8, mid, "encode")
    arrow(dr, P2[1] + 8, P3[0] - 8, mid, "decode")

    for xr, label in [(P1, "~200 ROIs over time"), (P2, "latent state space"),
                      (P3, "one-step-ahead forecast")]:
        text(dr, ((xr[0] + xr[1]) / 2, PB + 12), label, F_TINY,
             MUTED + (255,), center=True)

    text(dr, (W / 2, H - 46),
         "1-TR forecast MSE 0.268 vs AR(1) at 0.389 -- 31% better",
         F_BOLD, INK + (255,), center=True)
    text(dr, (W / 2, H - 26),
         "on 100% of held-out subjects, and 27% better on sites with an unseen TR",
         F_TINY, MUTED + (255,), center=True)

    return img.resize((W, H), Image.LANCZOS)


def main():
    frames = [frame(f) for f in range(F)]
    # One shared palette for every frame: the raster would otherwise push each
    # frame into its own 200+ colour table and triple the file size.
    ref = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=ref, dither=Image.NONE) for f in frames]

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "foundational-brain.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
