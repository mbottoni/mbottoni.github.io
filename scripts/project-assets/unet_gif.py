"""
Animated GIF for unet-fun (a from-scratch U-Net for image segmentation).

Draws the signature U-Net: an image enters the encoder (feature maps shrink as
they go down), passes through the bottleneck, and climbs the decoder (maps grow
back) to a segmentation mask. A pulse flows along that path and each skip
connection flashes as the pulse reaches the matching decoder level.

Regenerate:
  uv run --with pillow python scripts/project-assets/unet_gif.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 720, 330
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
ENC = (33, 86, 165)         # encoder (blue)
DEC = (186, 57, 37)         # decoder (red)
BOTTLE = (110, 80, 150)     # bottleneck (purple)
SKIP = (200, 200, 195)
SKIP_HOT = (224, 170, 70)
GRID = (228, 228, 224)

YS = [86, 130, 176, 222]            # shared encoder/decoder rows
ENC_X, DEC_X = 188, 470
HEIGHTS = [34, 27, 21, 16]          # feature-map "spatial size" per level
BW = 78
BOTTLE_C = (329, 270)

# pulse path: input -> encoder (down) -> bottleneck -> decoder (up) -> output
INPUT_C = (66, 86)
OUTPUT_C = (654, 86)
PATH = ([INPUT_C]
        + [(ENC_X, y) for y in YS]
        + [BOTTLE_C]
        + [(DEC_X, y) for y in reversed(YS)]
        + [OUTPUT_C])
SEG = [math.dist(PATH[i], PATH[i + 1]) for i in range(len(PATH) - 1)]
TOTAL = sum(SEG)


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
F_SM = load_font(12 * S)


def sc(p):
    return (p[0] * S, p[1] * S)


def point_at(d):
    for i, L in enumerate(SEG):
        if d <= L:
            t = d / L if L else 0
            return (PATH[i][0] + (PATH[i + 1][0] - PATH[i][0]) * t,
                    PATH[i][1] + (PATH[i + 1][1] - PATH[i][1]) * t)
        d -= L
    return PATH[-1]


def text_c(dr, xy, s, font, fill):
    bb = dr.textbbox((0, 0), s, font=font)
    dr.text(sc((xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))),
            s, font=font, fill=fill)


def block(dr, cx, cy, h, color, active):
    box = (cx - BW / 2, cy - h / 2, cx + BW / 2, cy + h / 2)
    if active:
        g = (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4)
        dr.rounded_rectangle([sc((g[0], g[1])), sc((g[2], g[3]))], radius=7 * S,
                             fill=color + (45,))
    fill = tuple(min(255, c + 150) for c in color) if not active else tuple(
        min(255, c + 90) for c in color)
    dr.rounded_rectangle([sc((box[0], box[1])), sc((box[2], box[3]))], radius=6 * S,
                         fill=fill + (255,), outline=color + (255,),
                         width=(3 if active else 2) * S)


def thumb(dr, cx, cy, mask):
    """3x3 mini image: mask=False -> colourful input, True -> binary mask."""
    cells = [
        [(210, 90, 70), (90, 150, 210), (200, 200, 120)],
        [(120, 190, 130), (210, 120, 180), (110, 160, 210)],
        [(200, 170, 90), (130, 200, 190), (190, 110, 150)],
    ]
    mask_cells = [[0, 1, 0], [1, 1, 1], [0, 1, 1]]
    s_ = 13
    x0, y0 = cx - 1.5 * s_, cy - 1.5 * s_
    for r in range(3):
        for c in range(3):
            col = (DEC if mask_cells[r][c] else (236, 236, 232)) if mask else cells[r][c]
            x, y = x0 + c * s_, y0 + r * s_
            dr.rectangle([sc((x + 1, y + 1)), sc((x + s_ - 1, y + s_ - 1))], fill=col + (255,))


def frame(d):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    px, py = point_at(d)

    text_c(dr, (W / 2, 22), "U-Net: encoder–decoder with skip connections for segmentation",
           F_CAP, MUTED + (255,))

    # skip connections (flash when the pulse reaches that decoder row)
    for i, y in enumerate(YS):
        hot = (px > 360 and abs(py - y) < 10)
        col = SKIP_HOT if hot else SKIP
        dr.line([sc((ENC_X + BW / 2, y)), sc((DEC_X - BW / 2, y))],
                fill=col + (255,), width=(3 if hot else 2) * S)

    # encoder / decoder blocks
    for i, y in enumerate(YS):
        block(dr, ENC_X, y, HEIGHTS[i], ENC, abs(px - ENC_X) < 44 and abs(py - y) < 24)
        block(dr, DEC_X, y, HEIGHTS[i], DEC, abs(px - DEC_X) < 44 and abs(py - y) < 24)
    # bottleneck
    block(dr, BOTTLE_C[0], BOTTLE_C[1], 14, BOTTLE,
          abs(px - BOTTLE_C[0]) < 50 and abs(py - BOTTLE_C[1]) < 20)
    text_c(dr, (BOTTLE_C[0], BOTTLE_C[1] + 22), "bottleneck", F_SM, MUTED + (255,))
    text_c(dr, (ENC_X, 250), "encoder ↓", F_SM, ENC + (255,))
    text_c(dr, (DEC_X, 250), "decoder ↑", F_SM, DEC + (255,))

    # input image / output mask thumbnails + labels
    thumb(dr, *INPUT_C, mask=False)
    thumb(dr, *OUTPUT_C, mask=True)
    text_c(dr, (INPUT_C[0], 124), "image", F_SM, MUTED + (255,))
    text_c(dr, (OUTPUT_C[0], 124), "mask", F_SM, MUTED + (255,))

    # pulse
    for r, a in ((12, 45), (8, 95)):
        dr.ellipse([sc((px - r, py - r)), sc((px + r, py + r))], fill=(40, 40, 40, a))
    dr.ellipse([sc((px - 4.5, py - 4.5)), sc((px + 4.5, py + 4.5))], fill=(30, 30, 30, 255))

    return img.resize((W, H), Image.LANCZOS)


def main():
    frames_n = 64
    step = TOTAL / frames_n
    frames = [frame(i * step) for i in range(frames_n)]
    frames += [frames[-1]] * 6      # hold on the produced mask
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "unet-fun.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
