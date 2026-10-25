"""Fine ground patches: where the graded ground is sharp (a cutting's wall, an abutment's slope under a bridge), the
2 m terrain draws a staircase of slanted cells (the user, 2026-10-25: "try increasing the terrain resolution
dynamically to get the walls and ground looking cleaner"). A terrain block has one grid size, so those cells are
holes in the terrain (the block, block_holes) and a 0.5 m mesh of the same graded ground fills them (the road tile,
tile_mesh: drawn and colliding, as the tunnels' and the railways').

Both sides find the cells alone, from the same heights on the same lattice (the terrain's samples at even metres of
the anchor frame), so they agree without talking: a 2 m cell is sharp where its corners lie SHARP_DZ_M or more apart
and the graded ground differs from the natural ground at one of them by GRADED_DZ_M (a man-made slope: natural cliffs
stay the terrain's), then grown by one cell (the slope's foot and top). A patch's rim is the terrain's own edge (its
heights along the border between two samples, linear as the terrain draws it), so no gap opens at the hole's edge.
"""
import os

import numpy as np

FINE_ON = os.environ.get("BNE_FINE_PATCH", "on") != "off"
SQUARE = 2.0                # m: the terrain's grid (heightmap.SQUARE)
STEP = 0.5                  # m: the patch's grid
SHARP_DZ_M = 1.5            # m between a cell's highest and lowest corner: a sharp cell ...
GRADED_DZ_M = 0.5           # ... where the grading moved the ground this much at one of its corners
MAX_CELLS = 2600            # per road tile: 25 points a cell, under the engine's 16-bit indices; more: the steepest kept
MATERIAL = "bne_tunnel_soil"


def lattice(x0, y0, x1, y1, margin=2):
    """The terrain samples (xs, ys) covering [x0, x1) x [y0, y1) and margin samples more around (the cells' growth)."""
    a = np.floor(x0 / SQUARE) - margin
    b = np.ceil(x1 / SQUARE) + margin
    c = np.floor(y0 / SQUARE) - margin
    d = np.ceil(y1 / SQUARE) + margin
    return np.arange(a, b + 1) * SQUARE, np.arange(c, d + 1) * SQUARE


def sharp_cells(u, raw):
    """(ny - 1, nx - 1) bool: the cells (between samples [j:j+2, i:i+2] of the heights u and the natural ground raw,
    both [j, i]) to draw finely: sharp and graded, grown by one cell. Pure."""
    c = np.stack([u[:-1, :-1], u[:-1, 1:], u[1:, :-1], u[1:, 1:]])
    dz = c.max(0) - c.min(0)
    moved = np.abs(u - raw)
    m = np.stack([moved[:-1, :-1], moved[:-1, 1:], moved[1:, :-1], moved[1:, 1:]]).max(0)
    sharp = (dz >= SHARP_DZ_M) & (m >= GRADED_DZ_M)
    grown = sharp.copy()
    grown[1:] |= sharp[:-1]
    grown[:-1] |= sharp[1:]
    g2 = grown.copy()
    g2[:, 1:] |= grown[:, :-1]
    g2[:, :-1] |= grown[:, 1:]
    return g2


def _heights(frame, elev, xs, ys):
    """Terrain heights (the frame's up, as heightmap.surface_u) at the points (xs, ys) (1-D, paired)."""
    la, lo, _ = frame.enu_to_geodetic(np.asarray(xs, np.float64), np.asarray(ys, np.float64), 0.0)
    la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
    return np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(elev(la, lo), np.float64))[2], np.float64)


def _grid(frame, fn, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return _heights(frame, fn, X.ravel(), Y.ravel()).reshape(X.shape)


def block_holes(u, raw):
    """(n, n) bool: the samples of a terrain block (heights u and natural ground raw [gy, gx]) that are holes for the
    fine patches (sharp_cells all round them). Pure."""
    out = np.zeros(u.shape, bool)
    if not FINE_ON:
        return out
    c = sharp_cells(u, raw)
    # a sample is a hole only where all four cells around it have a patch: the engine takes a hole sample's square
    # (which neighbour of it, in-game 2026-10-25: gaps where it was not the north-east one), and so it is always one
    # the patch covers (a ring of cells on the patch's edge draws both: the patch's rim on the terrain's own edge)
    out[1:-1, 1:-1] = c[1:, 1:] & c[:-1, 1:] & c[1:, :-1] & c[:-1, :-1]
    return out


def tile_mesh(frame, box, elev, raw):
    """The fine patch of road tile box (x0, y0, x1, y1) (the cells whose south-west sample lies in it): roadmesh.write's
    (positions, tris, uvs, materials, tri material) with positions relative to (x0, y0, 0), or None. elev(lat, lon):
    the graded ground (the terrain block's), raw: the natural ground."""
    if not FINE_ON:
        return None
    x0, y0, x1, y1 = box
    xs, ys = lattice(x0, y0, x1, y1)
    u = _grid(frame, elev, xs, ys)
    r = _grid(frame, raw, xs, ys)
    cells = sharp_cells(u, r)
    own = (xs[:-1][None, :] >= x0) & (xs[:-1][None, :] < x1) & (ys[:-1][:, None] >= y0) & (ys[:-1][:, None] < y1)
    keep = cells & own
    J, I = np.nonzero(keep)
    if not len(J):
        return None
    if len(J) > MAX_CELLS:
        c = np.stack([u[J, I], u[J, I + 1], u[J + 1, I], u[J + 1, I + 1]])
        order = np.argsort(-(c.max(0) - c.min(0)), kind="stable")[:MAX_CELLS]
        J, I = J[order], I[order]
    k = int(round(SQUARE / STEP))           # fine steps per cell side
    t = np.arange(k + 1) * STEP
    # per cell a (k+1)² grid of points
    px = xs[I][:, None, None] + t[None, None, :]
    py = ys[J][:, None, None] + t[None, :, None]
    PX = np.broadcast_to(px, (len(J), k + 1, k + 1))
    PY = np.broadcast_to(py, (len(J), k + 1, k + 1))
    z = _heights(frame, elev, PX.ravel(), PY.ravel()).reshape(len(J), k + 1, k + 1)
    # a border with a cell the terrain draws: its heights linear between the two samples, as the terrain's
    w = t / SQUARE
    nx, ny = cells.shape[1], cells.shape[0]

    def fine(jj, ii):
        return (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx) & cells[np.clip(jj, 0, ny - 1), np.clip(ii, 0, nx - 1)]
    s_w = ~fine(J, I - 1)
    s_e = ~fine(J, I + 1)
    s_s = ~fine(J - 1, I)
    s_n = ~fine(J + 1, I)
    u00, u10, u01, u11 = u[J, I], u[J, I + 1], u[J + 1, I], u[J + 1, I + 1]     # (x, y) corners
    z[s_s, 0, :] = (u00[s_s, None] * (1 - w) + u10[s_s, None] * w)
    z[s_n, k, :] = (u01[s_n, None] * (1 - w) + u11[s_n, None] * w)
    z[s_w, :, 0] = (u00[s_w, None] * (1 - w) + u01[s_w, None] * w)
    z[s_e, :, k] = (u10[s_e, None] * (1 - w) + u11[s_e, None] * w)
    z[:, 0, 0], z[:, 0, k], z[:, k, 0], z[:, k, k] = u00, u10, u01, u11
    P = np.stack([PX - x0, PY - y0, z], -1).reshape(-1, 3)
    base = (np.arange(len(J)) * (k + 1) ** 2)[:, None]
    a = (np.arange(k)[:, None] * (k + 1) + np.arange(k)[None, :]).ravel()       # each quad's south-west corner
    quads = np.stack([a, a + 1, a + k + 2, a + k + 1], 1)                      # sw, se, ne, nw
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])           # counter-clockwise from above
    T = (base[:, None, :] + tri[None, :, :]).reshape(-1, 3)
    UV = np.stack([PX.ravel() / 4.0, PY.ravel() / 4.0], 1)
    return P, T.astype(np.int64), UV, [MATERIAL], np.zeros(len(T), np.int64)
