"""
Animated GIF of the MLOps production lifecycle for the ml-prod project.

A pulse flows through the pipeline stages -- Data -> Train -> Docker -> K8s ->
Monitor -- lighting each up in turn; when Monitor detects drift it flashes and a
feedback arrow loops back to the start to retrain. Closed loop, so the GIF
animates seamlessly.

Regenerate:
  uv run --with pillow python scripts/project-assets/ml_prod_gif.py
"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 780, 250
S = 2                       # supersample factor
FRAMES = 72
HOLD = 0

BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
IDLE_BORDER = (205, 205, 200)
ACCENT = (186, 57, 37)      # red  -> active stage / pulse
ACCENT_SOFT = (245, 223, 216)
BLUE = (33, 86, 165)
AMBER = (200, 120, 30)      # drift warning
AMBER_SOFT = (250, 232, 205)

STAGES = ["Data", "Train", "Docker", "K8s", "Monitor"]


def load_font(size):
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = load_font(19 * S)
FONT_SM = load_font(14 * S)

# geometry (unscaled)
BOX_W, BOX_H = 112, 56
ROW_Y = 64
LOOP_Y = 196
centers = []
for i in range(len(STAGES)):
    left = 30 + i * (BOX_W + 40)
    centers.append((left + BOX_W / 2, ROW_Y + BOX_H / 2))

# closed pulse path: along the top row, then feedback back to the first stage
cy = ROW_Y + BOX_H / 2
path = [centers[i] for i in range(len(STAGES))]
path += [(centers[-1][0], LOOP_Y), (centers[0][0], LOOP_Y), (centers[0][0], cy)]

# precompute cumulative arc length for constant-speed motion
seglens = [math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
total = sum(seglens)


def point_at(d):
    d = d % total
    for i, L in enumerate(seglens):
        if d <= L:
            t = d / L if L else 0
            x = path[i][0] + (path[i + 1][0] - path[i][0]) * t
            y = path[i][1] + (path[i + 1][1] - path[i][1]) * t
            return x, y
        d -= L
    return path[-1]


def sc(p):
    return (p[0] * S, p[1] * S)


def rrect(dr, box, radius, fill, outline, width):
    dr.rounded_rectangle([sc((box[0], box[1])), sc((box[2], box[3]))],
                         radius=radius * S, fill=fill, outline=outline, width=width * S)


def arrow(dr, p0, p1, color, width):
    dr.line([sc(p0), sc(p1)], fill=color, width=width * S)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    a = 7
    for off in (math.radians(150), -math.radians(150)):
        x = p1[0] + a * math.cos(ang + off)
        y = p1[1] + a * math.sin(ang + off)
        dr.line([sc((x, y)), sc(p1)], fill=color, width=width * S)


def text_c(dr, xy, s, font, fill):
    bb = dr.textbbox((0, 0), s, font=font)
    dr.text(sc((xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))),
            s, font=font, fill=fill)


def frame(d):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    px, py = point_at(d)
    drift = px > centers[-1][0] - 20 and py < LOOP_Y - 10  # near Monitor / leaving it

    # connecting arrows along the top row
    for i in range(len(STAGES) - 1):
        x0 = centers[i][0] + BOX_W / 2
        x1 = centers[i + 1][0] - BOX_W / 2
        arrow(dr, (x0, cy), (x1, cy), MUTED + (255,), 2)

    # feedback loop (Monitor -> down -> left -> up into Data)
    fb = AMBER if drift else MUTED
    mx = centers[-1][0]
    dr.line([sc((mx, ROW_Y + BOX_H)), sc((mx, LOOP_Y))], fill=fb + (255,), width=2 * S)
    dr.line([sc((mx, LOOP_Y)), sc((centers[0][0], LOOP_Y))], fill=fb + (255,), width=2 * S)
    arrow(dr, (centers[0][0], LOOP_Y), (centers[0][0], ROW_Y + BOX_H), fb + (255,), 2)
    text_c(dr, ((mx + centers[0][0]) / 2, LOOP_Y + 16), "retrain on drift",
           FONT_SM, (AMBER if drift else MUTED) + (255,))

    # stage boxes
    for i, name in enumerate(STAGES):
        cx, ccy = centers[i]
        box = (cx - BOX_W / 2, ccy - BOX_H / 2, cx + BOX_W / 2, ccy + BOX_H / 2)
        active = math.dist((px, py), (cx, ccy)) < 46 and py < LOOP_Y - 20
        is_mon = (name == "Monitor")
        if active and is_mon and drift:
            rrect(dr, box, 10, AMBER_SOFT + (255,), AMBER + (255,), 3)
            tcol = AMBER
        elif active:
            # glow
            g = (box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6)
            rrect(dr, g, 13, ACCENT + (40,), None, 0)
            rrect(dr, box, 10, ACCENT_SOFT + (255,), ACCENT + (255,), 3)
            tcol = ACCENT
        else:
            rrect(dr, box, 10, (255, 255, 255, 255), IDLE_BORDER + (255,), 2)
            tcol = INK
        text_c(dr, (cx, ccy), name, FONT, tcol + (255,))

    if drift:
        text_c(dr, (centers[-1][0], ROW_Y - 14), "drift detected", FONT_SM, AMBER + (255,))

    # pulse (halo + core)
    for r, a in ((14, 45), (9, 90)):
        dr.ellipse([sc((px - r, py - r)), sc((px + r, py + r))], fill=ACCENT + (a,))
    dr.ellipse([sc((px - 5, py - 5)), sc((px + 5, py + 5))], fill=ACCENT + (255,))

    return img.resize((W, H), Image.LANCZOS)


def main():
    step = total / FRAMES
    frames = [frame(i * step) for i in range(FRAMES)]
    frames += [frames[-1]] * HOLD
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "ml-prod.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=85, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
