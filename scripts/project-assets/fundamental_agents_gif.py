"""
Animated GIF for fundamental-agents ("Stock Analyzer AI").

An orchestrator runs five specialized agents in sequence to turn a ticker into an
investment report. This animation walks a pulse down the agent pipeline: each
agent lights up and reveals its role as the orchestrator reaches it, and once all
five are done a report card appears. A placeholder ticker and the project's real
metric *names* are used (no fabricated figures about real companies).

Regenerate:
  uv run --with pillow python scripts/project-assets/fundamental_agents_gif.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 600, 484
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
ACCENT = (186, 57, 37)
GREEN = (31, 150, 90)
PEND = (210, 210, 205)
CARD = (255, 255, 255)

AGENTS = [
    ("Data Gathering", "financials · prices · news"),
    ("Financial Metrics", "P/E · D/E · ROE"),
    ("News Sentiment", "VADER on headlines"),
    ("Valuation", "DCF · WACC · FCF"),
    ("Synthesis", "markdown report"),
]

CARD_X0, CARD_X1 = 150, 470
CARD_H, GAP = 50, 12
TOP = 96
CENTERS = [TOP + i * (CARD_H + GAP) + CARD_H / 2 for i in range(len(AGENTS))]
RAIL_X = 116
REPORT_Y = TOP + len(AGENTS) * (CARD_H + GAP) + 8


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


F_NAME = load_font(15 * S, bold=True)
F_SUB = load_font(12 * S)
F_CAP = load_font(15 * S)
F_PILL = load_font(15 * S, bold=True)


def sc(p):
    return (p[0] * S, p[1] * S)


def text(dr, xy, s, font, fill, center=False):
    if center:
        bb = dr.textbbox((0, 0), s, font=font)
        xy = (xy[0] - (bb[2] - bb[0]) / (2 * S), xy[1] - (bb[3] - bb[1]) / (2 * S))
    dr.text(sc(xy), s, font=font, fill=fill)


def check(dr, cx, cy, color):
    dr.line([sc((cx - 5, cy)), sc((cx - 1, cy + 4))], fill=color + (255,), width=2 * S)
    dr.line([sc((cx - 1, cy + 4)), sc((cx + 6, cy - 4))], fill=color + (255,), width=2 * S)


def frame(py):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")

    text(dr, (W / 2, 22), "Multi-agent stock analysis: an orchestrator runs five agents",
         F_CAP, MUTED + (255,), center=True)

    # ticker input pill
    px0, px1, pyy = W / 2 - 60, W / 2 + 60, 56
    dr.rounded_rectangle([sc((px0, pyy - 15)), sc((px1, pyy + 15))], radius=15 * S,
                         fill=(238, 240, 245, 255), outline=(33, 86, 165, 255), width=2 * S)
    text(dr, (W / 2 - 12, pyy), "ACME", F_PILL, (33, 86, 165) + (255,), center=True)
    tx = W / 2 + 30
    dr.polygon([sc((tx, pyy - 5)), sc((tx, pyy + 5)), sc((tx + 8, pyy))],
               fill=(33, 86, 165, 255))

    # vertical rail (orchestrator)
    dr.line([sc((RAIL_X, pyy + 16)), sc((RAIL_X, CENTERS[-1]))], fill=PEND + (255,), width=2 * S)
    done_to = max((i for i, c in enumerate(CENTERS) if py >= c), default=-1)
    if done_to >= 0:
        dr.line([sc((RAIL_X, pyy + 16)), sc((RAIL_X, min(py, CENTERS[-1])))],
                fill=ACCENT + (255,), width=2 * S)

    all_done = done_to >= len(AGENTS) - 1 and py >= CENTERS[-1]

    for i, (name, sub) in enumerate(AGENTS):
        cy = CENTERS[i]
        done = py >= cy
        active = abs(py - cy) < 22 and not all_done
        # index node on the rail
        nodecol = ACCENT if done else PEND
        dr.ellipse([sc((RAIL_X - 9, cy - 9)), sc((RAIL_X + 9, cy + 9))],
                   fill=nodecol + (255,), outline=BG + (255,), width=2 * S)
        text(dr, (RAIL_X, cy - 1), str(i + 1), F_SUB,
             (255, 255, 255) + (255,), center=True)

        box = (CARD_X0, cy - CARD_H / 2, CARD_X1, cy + CARD_H / 2)
        if active:
            dr.rounded_rectangle([sc((box[0] - 4, box[1] - 4)), sc((box[2] + 4, box[3] + 4))],
                                 radius=10 * S, fill=ACCENT + (40,))
        border = ACCENT if (active or done) else PEND
        dr.rounded_rectangle([sc(box[:2]), sc(box[2:])], radius=8 * S,
                             fill=CARD + (255,), outline=border + (255,),
                             width=(3 if active else 2) * S)
        text(dr, (CARD_X0 + 16, cy - (12 if done else 6)), name, F_NAME,
             (INK if (done or active) else MUTED) + (255,))
        if done:
            text(dr, (CARD_X0 + 16, cy + 6), sub, F_SUB, MUTED + (255,))
            check(dr, CARD_X1 - 18, cy, GREEN)

    # pulse
    if not all_done:
        for r, a in ((11, 45), (7, 100)):
            dr.ellipse([sc((RAIL_X - r, py - r)), sc((RAIL_X + r, py + r))], fill=ACCENT + (a,))

    # report card (appears once every agent is done)
    if all_done:
        box = (CARD_X0, REPORT_Y, CARD_X1, REPORT_Y + 54)
        dr.rounded_rectangle([sc(box[:2]), sc(box[2:])], radius=8 * S,
                             fill=(244, 249, 245, 255), outline=GREEN + (255,), width=2 * S)
        text(dr, (CARD_X0 + 16, REPORT_Y + 11), "Investment report", F_NAME, INK + (255,))
        text(dr, (CARD_X0 + 16, REPORT_Y + 31), "metrics · sentiment · valuation · recommendation",
             F_SUB, MUTED + (255,))
        check(dr, CARD_X1 - 18, REPORT_Y + 27, GREEN)

    return img.resize((W, H), Image.LANCZOS)


def main():
    start, end = 50, CENTERS[-1] + 6
    n = 52
    seq = [start + (end - start) * i / (n - 1) for i in range(n)]
    frames = [frame(p) for p in seq]
    frames += [frame(end + 20)] * 12      # hold with the report shown
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "fundamental-agents.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=75, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} -> {out}")


if __name__ == "__main__":
    main()
