"""
Animated GIF for the Rubik's-cube RL solver (reinforcement-learning-final-project).

Renders an unfolded cube net that starts scrambled and solves itself, one move at
a time, by replaying the scramble in reverse -- exactly the "scramble, then walk
it back" idea the project uses to generate its training signal.

Each quarter-turn rotates a face's own 3x3 and cycles three stickers on each of
its four neighbours; every move is order-4 and invertible, so replaying the
inverse sequence is guaranteed to return to the solved cube (asserted below).

Regenerate:
  uv run --with pillow python scripts/project-assets/rubiks_solve_gif.py
"""
import os
import random

from PIL import Image, ImageDraw, ImageFont

S = 2
CELL = 30
FACE = 3 * CELL
GAP = 10
BG = (253, 253, 252)
LINE = (30, 30, 30)
MUTED = (120, 128, 140)

# face colours (U D F B L R)
COLORS = {
    "U": (245, 245, 245),   # white
    "D": (243, 209, 26),    # yellow
    "F": (31, 164, 91),     # green
    "B": (33, 86, 165),     # blue
    "L": (224, 123, 57),    # orange
    "R": (186, 57, 37),     # red
}
FACES = ["U", "D", "F", "B", "L", "R"]

# net layout (col, row) in face units:  cross with U over F, D under F
POS = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}

# clockwise rotation of a 3x3 face (row-major indices 0..8)
_ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2]

# For each move: the face to rotate, and four neighbour strips (face, [i,j,k]).
# strip[k] receives strip[(k+1) % 4] -> a 4-cycle (disjoint across faces).
MOVES = {
    "U": ("U", [("F", [0, 1, 2]), ("R", [0, 1, 2]), ("B", [0, 1, 2]), ("L", [0, 1, 2])]),
    "D": ("D", [("F", [6, 7, 8]), ("L", [6, 7, 8]), ("B", [6, 7, 8]), ("R", [6, 7, 8])]),
    "F": ("F", [("U", [6, 7, 8]), ("R", [0, 3, 6]), ("D", [2, 1, 0]), ("L", [8, 5, 2])]),
    "B": ("B", [("U", [2, 1, 0]), ("L", [0, 3, 6]), ("D", [6, 7, 8]), ("R", [8, 5, 2])]),
    "L": ("L", [("U", [0, 3, 6]), ("F", [0, 3, 6]), ("D", [0, 3, 6]), ("B", [8, 5, 2])]),
    "R": ("R", [("U", [8, 5, 2]), ("B", [0, 3, 6]), ("D", [8, 5, 2]), ("F", [8, 5, 2])]),
}


def solved_cube():
    return {f: [f] * 9 for f in FACES}


def rotate_cw(face):
    return [face[i] for i in _ROT]


def apply_move(cube, name):
    face, strips = MOVES[name]
    cube[face] = rotate_cw(cube[face])
    saved = [[cube[f][i] for i in idx] for f, idx in strips]
    for k, (f, idx) in enumerate(strips):
        src = saved[(k + 1) % 4]
        for pos, i in enumerate(idx):
            cube[f][i] = src[pos]
    return cube


def load_font(size):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


W, H = 4 * FACE + 3 * GAP + 80, 3 * FACE + 2 * GAP + 90
MARGIN_X = (W - (4 * FACE + 3 * GAP)) // 2
MARGIN_Y = 56
F_CAP = load_font(15 * S)


def sc(p):
    return (p[0] * S, p[1] * S)


def draw_cube(cube, highlight=None):
    img = Image.new("RGB", (W * S, H * S), BG)
    dr = ImageDraw.Draw(img)
    cap = "Solving a scrambled cube by walking the scramble backwards"
    bb = dr.textbbox((0, 0), cap, font=F_CAP)
    dr.text(sc((W / 2 - (bb[2] - bb[0]) / (2 * S), 22)), cap, font=F_CAP, fill=MUTED)
    for f in FACES:
        col, row = POS[f]
        fx = MARGIN_X + col * (FACE + GAP)
        fy = MARGIN_Y + row * (FACE + GAP)
        if highlight == f:
            dr.rounded_rectangle(
                [sc((fx - 4, fy - 4)), sc((fx + FACE + 4, fy + FACE + 4))],
                radius=8 * S, outline=LINE, width=2 * S,
            )
        for i in range(9):
            r, c = divmod(i, 3)
            x = fx + c * CELL
            y = fy + r * CELL
            dr.rounded_rectangle(
                [sc((x + 2, y + 2)), sc((x + CELL - 2, y + CELL - 2))],
                radius=5 * S, fill=COLORS[cube[f][i]], outline=LINE, width=max(1, S),
            )
    return img.resize((W, H), Image.LANCZOS)


def main():
    rng = random.Random(11)
    cube = solved_cube()
    scramble = [rng.choice(FACES) for _ in range(18)]
    for m in scramble:
        apply_move(cube, m)

    frames = [draw_cube(cube)] * 4          # hold on the scramble
    # solve = inverse scramble; inverse of an order-4 move is applying it 3x
    for m in reversed(scramble):
        for _ in range(3):
            apply_move(cube, m)
        frames.append(draw_cube(cube, highlight=MOVES[m][0]))
    frames += [draw_cube(cube)] * 8         # hold solved

    assert cube == solved_cube(), "cube did not return to solved!"

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects", "rl-rubiks-cube.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=170, loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)} solved={cube == solved_cube()} -> {out}")


if __name__ == "__main__":
    main()
