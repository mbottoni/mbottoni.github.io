"""
Animated GIF for vae-playground (a playground of Variational Autoencoder variants).

Shows the signature VAE picture: a structured 2D latent space where the encoder
has organised data into class clusters, a sampled point z traverses the space,
and the decoder turns z into a generated image that morphs as z moves. The
decoded panel is a smooth procedural stand-in (no trained decoder is run); the
latent path is periodic so the GIF loops.

Regenerate:
  uv run --with numpy --with pillow python scripts/project-assets/vae_latent_gif.py
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
PRIOR = (210, 210, 205)

PALETTE = [(33, 86, 165), (186, 57, 37), (31, 150, 90),
           (150, 110, 60), (120, 80, 150)]

rng = np.random.default_rng(1)
K = 5
PER = 44
centers = np.array([[1.8 * math.cos(2 * math.pi * k / K),
                     1.8 * math.sin(2 * math.pi * k / K)] for k in range(K)])
PTS = np.concatenate([centers[k] + rng.normal(0, 0.42, size=(PER, 2)) for k in range(K)])
LAB = np.concatenate([[k] * PER for k in range(K)])

LBOX = (60, 66, 432, 338)        # latent panel
DBOX = (520, 132, 660, 272)      # decoded panel (square)
LRANGE = 3.1


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


def sc(p):
    return (p[0] * S, p[1] * S)


def lat_to_px(pt):
    fx = (pt[0] / LRANGE + 1) / 2
    fy = (pt[1] / LRANGE + 1) / 2
    x = LBOX[0] + fx * (LBOX[2] - LBOX[0])
    y = LBOX[3] - fy * (LBOX[3] - LBOX[1])
    return x, y


def text_c(dr, xy, s, font, fill):
    bb = dr.textbbox((0, 0), s, font=font)
    dr.text(sc((xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))),
            s, font=font, fill=fill)


def decode(z, grid=16):
    """Procedural stand-in for a decoder: a smooth field whose blobs move with z."""
    ax = np.linspace(-1, 1, grid)
    gx, gy = np.meshgrid(ax, ax)
    c1 = np.clip(z / LRANGE, -1, 1)
    c2 = -0.6 * c1[::-1]
    img = (np.exp(-((gx - c1[0]) ** 2 + (gy - c1[1]) ** 2) / 0.18)
           + 0.8 * np.exp(-((gx - c2[0]) ** 2 + (gy - c2[1]) ** 2) / 0.5)
           + 0.5 * np.exp(-((gx + c1[1]) ** 2 + (gy - c1[0]) ** 2) / 0.3))
    img = img / img.max()
    return np.round(img * 9) / 9        # quantize -> small GIF palette


def frame(t):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    text_c(dr, (W / 2, 24),
           "A VAE learns a smooth latent space — sample z, the decoder generates an image",
           F_CAP, MUTED + (255,))

    # latent panel frame + prior reference circle
    dr.rounded_rectangle([sc(LBOX[:2]), sc(LBOX[2:])], radius=8 * S,
                         outline=(222, 222, 218, 255), width=S, fill=(250, 250, 248, 255))
    cx, cy = lat_to_px((0, 0))
    rpx = (LBOX[2] - LBOX[0]) / 2 / LRANGE
    for rr in (1, 2):
        dr.ellipse([sc((cx - rr * rpx, cy - rr * rpx)), sc((cx + rr * rpx, cy + rr * rpx))],
                   outline=PRIOR + (255,), width=S)
    text_c(dr, ((LBOX[0] + LBOX[2]) / 2, LBOX[3] + 14), "latent space (2D)", F_SM, MUTED + (255,))

    # encoded points, coloured by class
    for i in range(len(PTS)):
        x, y = lat_to_px(PTS[i])
        col = PALETTE[LAB[i]]
        dr.ellipse([sc((x - 2.4, y - 2.4)), sc((x + 2.4, y + 2.4))], fill=col + (185,))

    # sampled z on a smooth periodic path
    z = np.array([2.1 * math.sin(2 * math.pi * t), 2.1 * math.sin(2 * math.pi * 2 * t + 1.0)])
    zx, zy = lat_to_px(z)
    for r, a in ((11, 50), (7, 110)):
        dr.ellipse([sc((zx - r, zy - r)), sc((zx + r, zy + r))], fill=(30, 30, 30, a))
    dr.ellipse([sc((zx - 4.5, zy - 4.5)), sc((zx + 4.5, zy + 4.5))], fill=(20, 20, 20, 255))
    text_c(dr, (zx, zy - 16), "z", F_SM, INK + (255,))

    # decode arrow
    ax0 = LBOX[2] + 12
    dr.line([sc((ax0, (DBOX[1] + DBOX[3]) / 2)), sc((DBOX[0] - 12, (DBOX[1] + DBOX[3]) / 2))],
            fill=MUTED + (255,), width=2 * S)
    text_c(dr, ((ax0 + DBOX[0]) / 2, (DBOX[1] + DBOX[3]) / 2 - 14), "decode", F_SM, MUTED + (255,))

    # decoded image
    g = 16
    field = decode(z, g)
    cw = (DBOX[2] - DBOX[0]) / g
    for r in range(g):
        for c in range(g):
            v = int(245 - 220 * field[r, c])
            x = DBOX[0] + c * cw
            y = DBOX[1] + r * cw
            dr.rectangle([sc((x, y)), sc((x + cw, y + cw))], fill=(v, v, v, 255))
    dr.rounded_rectangle([sc(DBOX[:2]), sc(DBOX[2:])], radius=4 * S,
                         outline=(200, 200, 196, 255), width=S)
    text_c(dr, ((DBOX[0] + DBOX[2]) / 2, DBOX[3] + 14), "decoded sample", F_SM, MUTED + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    n = 50
    frames = [frame(i / n) for i in range(n)]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "vae-playground.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=75, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
