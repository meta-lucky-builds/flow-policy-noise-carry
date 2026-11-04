"""2-D point-mass diagnostic of spec section 8.2. Batched where it matters."""
import numpy as np

DT = 0.05
V_MAX = 1.0
U_MAX = 2.0
OBST_R = 0.2
GOAL_TOL = 0.1
GOAL_SPEED = 0.2
T_MAX = 200

# Allowed neural view. No commanded or executed action vector appears here.
OBS_FIELDS = ("pos_x", "pos_y", "vel_x", "vel_y", "goal_x", "goal_y",
              "obst_x", "obst_y", "nuis_0", "nuis_1")
OBS_DIM = len(OBS_FIELDS)


def step(x, v, u):
    """x_{t+1}=x_t+dt v_t; v_{t+1}=clip(v_t+dt u_t). Works on any broadcastable shape."""
    u = np.clip(u, -U_MAX, U_MAX)
    return x + DT * v, np.clip(v + DT * u, -V_MAX, V_MAX)


def seg_point_dist(a, b, o):
    """Distance from point(s) o to segment(s) a-b, for the swept collision check."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1, keepdims=True), 1e-18)
    s = np.clip(((o - a) * ab).sum(-1, keepdims=True) / denom, 0.0, 1.0)
    return np.linalg.norm(a + s * ab - o, axis=-1)


def sample_scene(rng):
    return dict(
        start=np.array([-0.8, 0.0]) + rng.uniform(-0.05, 0.05, 2),
        goal=np.array([0.8, 0.0]) + rng.uniform(-0.05, 0.05, 2),
        obst=rng.uniform(-0.05, 0.05, 2),
        nuis=rng.normal(size=2),
    )


def observe(x, v, goal, obst, nuis):
    return np.concatenate([x, v, goal, obst, nuis], axis=-1)


class BatchEnv:
    """Parallel scenes. Each slot terminates on success, collision, leaving the workspace or T_MAX."""

    def __init__(self, seeds):
        scenes = [sample_scene(np.random.default_rng(s)) for s in seeds]
        self.x = np.stack([s["start"] for s in scenes])
        self.v = np.zeros_like(self.x)
        self.goal = np.stack([s["goal"] for s in scenes])
        self.obst = np.stack([s["obst"] for s in scenes])
        self.nuis = np.stack([s["nuis"] for s in scenes])
        n = len(seeds)
        self.t = 0
        self.done = np.zeros(n, bool)
        self.success = np.zeros(n, bool)
        self.collided = np.zeros(n, bool)
        self.finish_t = np.full(n, T_MAX)
        self.route = np.zeros(n, int)  # +1 passes above obstacle, -1 below, 0 not yet
        self.side_changes = np.zeros(n, int)
        self._side = np.zeros(n, int)

    def obs(self):
        return observe(self.x, self.v, self.goal, self.obst, self.nuis)

    def step(self, u):
        active = ~self.done
        nx, nv = step(self.x, self.v, u)
        hit = seg_point_dist(self.x, nx, self.obst) < OBST_R
        crossed = (self.x[:, 0] < self.obst[:, 0]) & (nx[:, 0] >= self.obst[:, 0]) & (self.route == 0)
        self.route[crossed & active] = np.sign(nx[crossed & active, 1] - self.obst[crossed & active, 1]).astype(int)
        # Dither measure: lateral side changes while still approaching the obstacle.
        lat = nx - self.obst
        side = np.where(np.abs(lat[:, 1]) > 0.03, np.sign(lat[:, 1]), 0).astype(int)
        approaching = nx[:, 0] < self.obst[:, 0]
        change = active & approaching & (side != 0) & (self._side != 0) & (side != self._side)
        self.side_changes += change
        self._side = np.where(side != 0, side, self._side)

        self.x = np.where(active[:, None], nx, self.x)
        self.v = np.where(active[:, None], nv, self.v)
        self.t += 1
        out = np.abs(self.x).max(1) > 1.0
        ok = (np.linalg.norm(self.x - self.goal, axis=1) <= GOAL_TOL) & (np.linalg.norm(self.v, axis=1) <= GOAL_SPEED)
        self.collided |= active & (hit | out)
        self.success |= active & ok & ~self.collided
        newly = active & (self.collided | self.success | (self.t >= T_MAX))
        self.finish_t[newly] = self.t
        self.done |= newly
