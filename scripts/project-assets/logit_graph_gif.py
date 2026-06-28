"""
Animated GIF of a *temporal logit graph* growing over time (synthetic).

Temporal model flavor: at each step a new node arrives and attaches to existing
nodes with probability driven by their degree --
    logit(P[attach to j]) ~ sigma + alpha * degree_j
i.e. preferential (rich-get-richer) growth, the temporal variant of the
logit-graph model. We use sub-linear weights so hubs emerge without collapsing
into a single star. Nodes hold fixed (force-directed) positions and edges appear
over time, so the animation reads as the network filling in.
"""
import math
import numpy as np
from PIL import Image, ImageDraw

rng = np.random.default_rng(3)

N = 55                 # final node count
SEED_NODES = 3         # initial connected core
GAMMA = 0.75           # sub-linear preferential-attachment exponent
W, H = 680, 470        # canvas
PAD = 42
HOLD = 10              # end frames holding the final graph

BG = (253, 253, 252)
EDGE = (120, 130, 150)
LOW = (33, 86, 165)        # low-degree node  (blue)
HIGH = (186, 57, 37)       # high-degree node (red)


def simulate():
    """Temporal preferential growth. Returns (steps, adjacency).
    steps[t] = (new_edges, degree_vector, active_mask)."""
    deg = np.zeros(N)
    adj = np.zeros((N, N), dtype=bool)
    active = np.zeros(N, dtype=bool)

    for a in range(SEED_NODES):
        active[a] = True
        for b in range(a + 1, SEED_NODES):
            adj[a, b] = adj[b, a] = True
            deg[a] += 1
            deg[b] += 1

    steps = [([], deg.copy(), active.copy())]
    for i in range(SEED_NODES, N):
        active[i] = True
        live = np.arange(i)
        w = (deg[live] + 1.0) ** GAMMA          # sub-linear preferential weight
        probs = w / w.sum()
        m = min(int(rng.choice([1, 2, 2, 3])), i)
        targets = rng.choice(i, size=m, replace=False, p=probs)
        new = []
        for j in map(int, targets):
            adj[i, j] = adj[j, i] = True
            deg[i] += 1
            deg[j] += 1
            new.append((i, j))
        steps.append((new, deg.copy(), active.copy()))
    return steps, adj


def fr_layout(adj, iters=320):
    """Fruchterman-Reingold: edges attract, all nodes repel."""
    n = len(adj)
    pos = rng.normal(0, 1, (n, 2))
    k = 1.3 / math.sqrt(n)
    ei, ej = np.nonzero(np.triu(adj))
    t = 0.1
    for _ in range(iters):
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos
            dist = np.hypot(delta[:, 0], delta[:, 1])
            dist[i] = 1.0
            f = (k * k) / (dist * dist)
            disp[i] = (delta.T * f).T.sum(axis=0)
        for a, b in zip(ei, ej):
            delta = pos[a] - pos[b]
            dist = math.hypot(delta[0], delta[1]) + 1e-9
            f = dist / k
            d = delta * f
            disp[a] -= d
            disp[b] += d
        length = np.hypot(disp[:, 0], disp[:, 1]) + 1e-9
        step = np.minimum(length, t)
        pos += (disp.T / length * step).T
        t = max(t * 0.985, 0.008)
    mn, mx = pos.min(0), pos.max(0)
    pos = (pos - mn) / (mx - mn + 1e-9)
    pos[:, 0] = PAD + pos[:, 0] * (W - 2 * PAD)
    pos[:, 1] = PAD + pos[:, 1] * (H - 2 * PAD)
    return pos


def lerp(c0, c1, t):
    return tuple(int(round(c0[k] + (c1[k] - c0[k]) * t)) for k in range(3))


def draw(pos, edges, deg, active, dmax, new_edges=(), scale=2):
    img = Image.new("RGB", (W * scale, H * scale), BG)
    dr = ImageDraw.Draw(img, "RGBA")
    newset = {tuple(sorted(e)) for e in new_edges}
    for (i, j) in edges:
        if tuple(sorted((i, j))) in newset:
            col, wid = (186, 57, 37, 235), 2 * scale
        else:
            a = 55 + int(125 * (min(deg[i], deg[j]) / dmax))
            col, wid = EDGE + (a,), max(1, scale)
        dr.line([tuple(pos[i] * scale), tuple(pos[j] * scale)], fill=col, width=wid)
    for v in range(N):
        if not active[v]:
            continue
        frac = (deg[v] / dmax) ** 0.7
        r = (3.2 + 10.0 * frac) * scale
        x, y = pos[v] * scale
        dr.ellipse([x - r, y - r, x + r, y + r], fill=lerp(LOW, HIGH, frac) + (255,),
                   outline=BG + (255,), width=max(1, scale // 2))
    return img.resize((W, H), Image.LANCZOS)


def main():
    steps, adj = simulate()
    pos = fr_layout(adj)
    dmax = max(steps[-1][1].max(), 1)
    frames, edges = [], []
    for (new, deg, active) in steps:
        edges += new
        frames.append(draw(pos, edges, deg, active, dmax, new_edges=new))
    frames += [frames[-1]] * HOLD
    # Repo-relative output. Regenerate with:
    #   uv run --with numpy --with pillow python scripts/project-assets/logit_graph_gif.py
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "content", "assets", "projects",
                       "logit-graph.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=110, loop=0, optimize=True, disposal=2)
    deg = steps[-1][1]
    print(f"frames={len(frames)} nodes={N} edges={len(edges)} "
          f"max_deg={int(deg.max())} mean_deg={deg.mean():.1f}")


if __name__ == "__main__":
    main()
