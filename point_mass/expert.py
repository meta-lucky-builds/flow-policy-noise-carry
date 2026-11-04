"""A* corridor plus convex QP expert (spec 8.2). Every accepted trajectory is forward-simulated."""
import heapq
import numpy as np
import osqp
import scipy.sparse as sp

from env import DT, V_MAX, U_MAX, OBST_R, step, seg_point_dist

GRID = 0.04
INFLATE = 0.05
MARGIN = 0.03
_coords = np.arange(-1.0, 1.0 + 1e-9, GRID)


def _segment_ok(a, b, obst, wall):
    if seg_point_dist(a, b, obst) < OBST_R + INFLATE - 1e-6:
        return False
    if wall and (a[0] - obst[0]) * (b[0] - obst[0]) < 0:
        s = (obst[0] - a[0]) / (b[0] - a[0])
        y = a[1] + s * (b[1] - a[1])
        if (wall > 0 and y < obst[1]) or (wall < 0 and y > obst[1]):
            return False
    return True


def astar(start, goal, obst, route):
    """route=+1 forces passing above the obstacle, -1 below, 0 no constraint. Returns waypoints or None."""
    n = len(_coords)
    xx, yy = np.meshgrid(_coords, _coords, indexing="ij")
    free = np.hypot(xx - obst[0], yy - obst[1]) >= OBST_R + INFLATE
    if route:
        col = np.abs(xx - obst[0]) <= GRID
        free &= ~(col & ((yy < obst[1]) if route > 0 else (yy > obst[1])))

    def nearest(p):
        d = np.hypot(xx - p[0], yy - p[1])
        d[~free] = np.inf
        return np.unravel_index(np.argmin(d), d.shape)

    s, g = nearest(start), nearest(goal)
    gpos = np.array([_coords[g[0]], _coords[g[1]]])
    best = {s: 0.0}
    parent = {}
    pq = [(0.0, s)]
    moves = [(i, j, np.hypot(i, j) * GRID) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    while pq:
        _, c = heapq.heappop(pq)
        if c == g:
            break
        for i, j, w in moves:
            m = (c[0] + i, c[1] + j)
            if not (0 <= m[0] < n and 0 <= m[1] < n) or not free[m]:
                continue
            ng = best[c] + w
            if ng < best.get(m, np.inf):
                best[m] = ng
                parent[m] = c
                heapq.heappush(pq, (ng + np.hypot(_coords[m[0]] - gpos[0], _coords[m[1]] - gpos[1]), m))
    if g not in best:
        return None
    path = [g]
    while path[-1] != s:
        path.append(parent[path[-1]])
    pts = [np.asarray(start)] + [np.array([_coords[a], _coords[b]]) for a, b in reversed(path)] + [np.asarray(goal)]
    # Line-of-sight shortcutting, so the corridor is not grid-jagged.
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_ok(pts[i], pts[j], obst, route):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def resample(path, N):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], N + 1)
    return np.stack([np.interp(q, s, path[:, i]) for i in range(2)], axis=1)


def solve_qp(x0, v0, goal, obst, w):
    """Decision vector: x_0..x_N, v_0..v_N, u_0..u_{N-1}. Spec cost, limits and tangent half-spaces."""
    N = len(w) - 1
    nx = 2 * (N + 1)
    ix, iv, iu = 0, nx, 2 * nx
    nvar = 2 * nx + 2 * N

    diag = np.zeros(nvar)
    diag[ix:ix + nx] = 2.0
    diag[iu:iu + 2 * N] = 2 * 0.01
    D = sp.diags([-np.ones(2 * N - 2), np.ones(2 * N - 2)], [0, 2], shape=(2 * N - 2, 2 * N))
    Pu = 2 * 0.05 * (D.T @ D)
    P = sp.diags(diag).tocsc() + sp.bmat([[sp.csc_matrix((iu, iu)), None], [None, Pu]]).tocsc()
    q = np.zeros(nvar)
    q[ix:ix + nx] = -2 * w.ravel()

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, l, h):
        nonlocal r
        for c, val in entries:
            rows.append(r); cols.append(c); vals.append(val)
        lo.append(l); hi.append(h)
        r += 1

    for k in range(2):
        add([(ix + k, 1.0)], x0[k], x0[k])
        add([(iv + k, 1.0)], v0[k], v0[k])
    for t in range(N):
        for k in range(2):
            add([(ix + 2 * (t + 1) + k, 1.0), (ix + 2 * t + k, -1.0), (iv + 2 * t + k, -DT)], 0.0, 0.0)
            add([(iv + 2 * (t + 1) + k, 1.0), (iv + 2 * t + k, -1.0), (iu + 2 * t + k, -DT)], 0.0, 0.0)
    for t in range(N + 1):
        for k in range(2):
            lx, hx = -1.0, 1.0
            lv, hv = -V_MAX, V_MAX
            if t == N:
                lx, hx = goal[k] - 0.05, goal[k] + 0.05
                lv, hv = -0.1, 0.1
            add([(ix + 2 * t + k, 1.0)], lx, hx)
            add([(iv + 2 * t + k, 1.0)], lv, hv)
    for i in range(2 * N):
        add([(iu + i, 1.0)], -U_MAX, U_MAX)
    for t in range(N + 1):
        d = w[t] - obst
        nrm = np.linalg.norm(d)
        if nrm < 1e-6:
            return None
        n = d / nrm
        add([(ix + 2 * t, n[0]), (ix + 2 * t + 1, n[1])], OBST_R + MARGIN + n @ obst, np.inf)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(r, nvar))
    prob = osqp.OSQP()
    prob.setup(P=sp.triu(P).tocsc(), q=q, A=A, l=np.array(lo), u=np.array(hi), verbose=False,
               eps_abs=1e-6, eps_rel=1e-6, max_iter=40000, polishing=True)
    res = prob.solve()
    if res.x is None or "solved" not in str(res.info.status).lower() or "inaccurate" in str(res.info.status).lower():
        return None
    return res.x[iu:iu + 2 * N].reshape(N, 2)


def verify(x0, v0, goal, obst, U):
    """Forward-simulate with the true integrator. Returns (ok, xs, vs)."""
    xs, vs = [np.asarray(x0)], [np.asarray(v0)]
    for u in U:
        nx, nv = step(xs[-1], vs[-1], u)
        if seg_point_dist(xs[-1], nx, obst) < OBST_R or np.abs(nx).max() > 1.0:
            return False, None, None
        xs.append(nx); vs.append(nv)
    ok = np.abs(xs[-1] - goal).max() <= 0.05 + 1e-4 and np.abs(vs[-1]).max() <= 0.1 + 1e-4
    return ok, np.array(xs), np.array(vs)


def plan(x0, v0, goal, obst, route, N):
    """Returns (U, xs, vs) or None. None means 'not certified by planner', never 'infeasible'."""
    path = astar(x0, goal, obst, route)
    if path is None:
        return None
    U = solve_qp(x0, v0, goal, obst, resample(path, N))
    if U is None:
        return None
    U = np.clip(U, -U_MAX, U_MAX)
    ok, xs, vs = verify(x0, v0, goal, obst, U)
    return (U, xs, vs) if ok else None
