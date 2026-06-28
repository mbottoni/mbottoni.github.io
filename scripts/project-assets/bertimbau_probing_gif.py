"""
Animated GIF for the BERTimbau probing project (ep2-nlp).

The project fine-tunes BERTimbau (Portuguese BERT) on B2W product reviews and
probes whether its embeddings can recover a surface statistic of the text: the
*vowel density* (share of letters that are vowels). This animation depicts that
task: a scan sweeps across a Portuguese review, vowels light up, and a gauge --
labelled as BERTimbau's prediction -- converges to the running vowel density.
Cycles through a few reviews and loops.

Regenerate:
  uv run --with pillow python scripts/project-assets/bertimbau_probing_gif.py
"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 760, 230
S = 2
BG = (253, 253, 252)
INK = (40, 40, 40)
MUTED = (120, 128, 140)
ACCENT = (186, 57, 37)      # vowels / fill
BLUE = (33, 86, 165)
TRACK = (228, 228, 224)

PHRASES = [
    "entrega rápida e ótima qualidade",
    "produto excelente, recomendo muito",
    "não gostei, veio com defeito",
]
VOWELS = set("aeiouáàâãéêíóôõúüAEIOUÁÀÂÃÉÊÍÓÔÕÚÜ")

SCAN_FRAMES = 26
HOLD_FRAMES = 8


def load_font(size, bold=False):
    names = ([
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
    ])
    for p in names:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_PHRASE = load_font(30 * S, bold=True)
F_LABEL = load_font(15 * S)
F_SMALL = load_font(13 * S)
F_PILL = load_font(15 * S, bold=True)


def sc(p):
    return (p[0] * S, p[1] * S)


def density_of(s):
    letters = [c for c in s if c.isalpha()]
    v = sum(1 for c in letters if c in VOWELS)
    return v / max(len(letters), 1)


def frame(dr, phrase, scan_t):
    """scan_t in [0,1] over the phrase width."""
    # char x positions (unscaled), centered
    widths = [F_PHRASE.getlength(c) / S for c in phrase]
    total = sum(widths)
    x = (W - total) / 2
    phrase_y = 86
    scan_x = x + scan_t * total

    letters_seen = vowels_seen = 0
    cx = x
    for c, w in zip(phrase, widths):
        passed = (cx + w / 2) < scan_x
        is_v = c in VOWELS
        if c.isalpha() and passed:
            letters_seen += 1
            if is_v:
                vowels_seen += 1
        col = ACCENT if (is_v and passed) else (INK if passed else MUTED)
        dr.text(sc((cx, phrase_y)), c, font=F_PHRASE, fill=col + (255,))
        cx += w

    # scan line
    if scan_t < 1.0:
        dr.line([sc((scan_x, phrase_y - 6)), sc((scan_x, phrase_y + 40))],
                fill=BLUE + (120,), width=2 * S)

    running = vowels_seen / max(letters_seen, 1)

    # BERTimbau pill -> gauge row
    row_y = 168
    pill = (x_pill0, y0, x_pill1, y1) = (W / 2 - 232, row_y - 18, W / 2 - 92, row_y + 18)
    dr.rounded_rectangle([sc((x_pill0, y0)), sc((x_pill1, y1))], radius=18 * S,
                         fill=(238, 240, 245, 255), outline=BLUE + (255,), width=2 * S)
    tb = dr.textbbox((0, 0), "BERTimbau", font=F_PILL)
    dr.text(sc(((x_pill0 + x_pill1) / 2 - (tb[2] - tb[0]) / (2 * S),
                row_y - (tb[3] - tb[1]) / (2 * S) - 1)), "BERTimbau",
            font=F_PILL, fill=BLUE + (255,))

    # arrow
    dr.line([sc((x_pill1 + 6, row_y)), sc((x_pill1 + 30, row_y))], fill=MUTED + (255,), width=2 * S)
    for off in (math.radians(150), -math.radians(150)):
        ax = x_pill1 + 30 + 7 * math.cos(off)
        ay = row_y + 7 * math.sin(off)
        dr.line([sc((ax, ay)), sc((x_pill1 + 30, row_y))], fill=MUTED + (255,), width=2 * S)

    # gauge (track + fill); full width maps to density 0..0.6
    gx0, gx1 = x_pill1 + 40, x_pill1 + 250
    gy0, gy1 = row_y - 11, row_y + 11
    dr.rounded_rectangle([sc((gx0, gy0)), sc((gx1, gy1))], radius=11 * S, fill=TRACK + (255,))
    frac = min(running / 0.6, 1.0)
    fillw = (gx1 - gx0) * frac
    if fillw > 4:
        dr.rounded_rectangle([sc((gx0, gy0)), sc((gx0 + fillw, gy1))], radius=11 * S,
                             fill=ACCENT + (255,))
    dr.text(sc((gx1 + 12, row_y - 11)), f"{running * 100:.0f}%",
            font=F_LABEL, fill=ACCENT + (255,))
    dr.text(sc((gx0, gy1 + 8)), "predicted vowel density", font=F_SMALL, fill=MUTED + (255,))

    # caption
    cap = "Probing BERTimbau — recovering vowel density from review text"
    cb = dr.textbbox((0, 0), cap, font=F_LABEL)
    dr.text(sc((W / 2 - (cb[2] - cb[0]) / (2 * S), 26)), cap, font=F_LABEL, fill=MUTED + (255,))


def render(phrase, scan_t):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    frame(dr, phrase, scan_t)
    return img.resize((W, H), Image.LANCZOS)


def main():
    frames = []
    for phrase in PHRASES:
        for i in range(SCAN_FRAMES):
            t = (i + 1) / SCAN_FRAMES
            # ease-out
            t = 1 - (1 - t) ** 2
            frames.append(render(phrase, t))
        frames += [render(phrase, 1.0)] * HOLD_FRAMES
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "bertimbau-probing.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} densities=" +
          ", ".join(f"{density_of(p)*100:.0f}%" for p in PHRASES) + f" -> {out}")


if __name__ == "__main__":
    main()
