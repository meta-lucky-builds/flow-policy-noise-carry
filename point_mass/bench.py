"""Closed-loop benchmark rollouts. Scenes and events depend only on the seed, so methods are paired."""
import numpy as np
import torch

from env import BatchEnv, DT, OBST_R, T_MAX, seg_point_dist, step
from stage0 import adapt, solve
from wrapper import carry_source

H, D, STEPS = 16, 2, 10
CONDITIONS = ("clean", "goal", "obstacle", "velocity", "delay2")
ENSEMBLE_M = 0.01


class EventEnv(BatchEnv):
    """BatchEnv plus one seeded event per scene. Unsafe obstacle moves become no-ops and are counted."""

    def __init__(self, seeds, condition):
        super().__init__(seeds)
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        self.delay = 2 if condition == "delay2" else 0
        n = len(seeds)
        self.te = np.zeros(n, int)
        self.mag, self.ang, self.jit = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, s in enumerate(seeds):
            r = np.random.default_rng(int(s) + 7_000_000)
            self.te[i] = r.integers(20, 51)
            self.mag[i] = r.uniform(0.1, 0.3)
            self.ang[i] = r.uniform(0, 2 * np.pi)
            self.jit[i] = r.uniform(-0.05, 0.05)
        self.event_noop = np.zeros(n, bool)
        self._hist = []

    def record_tick(self):
        """Keeps one observation per environment tick, so delay counts ticks and not replans."""
        self._hist.append(self.obs())

    def current_policy_obs(self):
        return self._hist[max(0, len(self._hist) - 1 - self.delay)]

    def _apply_events(self):
        hit = (self.te == self.t) & ~self.done
        if not hit.any():
            return
        c = np.stack([np.cos(self.ang), np.sin(self.ang)], axis=1) * self.mag[:, None]
        if self.condition == "goal":
            self.goal[hit] = np.clip(self.goal[hit] + c[hit], [0.5, -0.6], [0.95, 0.6])
        elif self.condition == "velocity":
            self.v[hit] = np.clip(self.v[hit] + c[hit], -1, 1)
        elif self.condition == "obstacle":
            side = np.sign(self.x[:, 1] - self.obst[:, 1])
            side[side == 0] = 1
            new = self.obst + np.stack([self.jit, side * self.mag], axis=1)
            safe = np.linalg.norm(self.x - new, axis=1) >= OBST_R + 0.08
            apply = hit & safe
            self.obst[apply] = new[apply]
            self.event_noop |= hit & ~safe

    def step(self, u):
        self._apply_events()
        super().step(u)


@torch.no_grad()
def rollout(policy, seeds, condition="clean", L=4, rho=0.0, mode="shift", ensemble=False, gen_seed=0):
    gen = torch.Generator().manual_seed(gen_seed)
    e = EventEnv(seeds, condition)
    n = len(seeds)
    z = torch.randn(n, H, D, generator=gen)
    e.record_tick()
    seam_sq, seam_n, du_sq, du_n = 0.0, 0, 0.0, 0
    prev_act, prev_u = None, np.zeros((n, D))
    has_prev = np.zeros(n, bool)
    if ensemble:
        psum, wsum = np.zeros((n, T_MAX + H, D)), np.zeros((n, T_MAX + H))
    first = True
    while not e.done.all():
        idx = np.flatnonzero(~e.done)
        if ensemble:
            z = torch.randn(n, H, D, generator=gen)
        elif not first:
            z = carry_source(z, rho, L, gen, mode=mode)
        ob = torch.from_numpy(e.current_policy_obs()[idx]).float()
        act = adapt(policy, solve(policy, z[idx], ob)).numpy()
        if prev_act is not None and not ensemble:
            d = prev_act[idx, L - 1] - act[:, 0]
            seam_sq += float((d ** 2).sum()); seam_n += d.shape[0]
        if prev_act is None:
            prev_act = np.zeros((n, H, D))
        prev_act[idx] = act
        first = False
        if ensemble:
            w = np.exp(-ENSEMBLE_M * e.t)
            psum[idx[:, None], e.t + np.arange(H)] += w * act
            wsum[idx[:, None], e.t + np.arange(H)] += w
            exec_steps = [psum[np.arange(n), e.t] / np.maximum(wsum[np.arange(n), e.t], 1e-12)[:, None]]
        else:
            exec_steps = [prev_act[:, j] for j in range(L)]
        for u in exec_steps:
            act_now = ~e.done
            du = u - prev_u
            mask = act_now & has_prev
            du_sq += float((du[mask] ** 2).sum()); du_n += int(mask.sum())
            prev_u = u.copy()
            has_prev |= act_now
            e.step(u)
            e.record_tick()
            if e.done.all():
                break
    return dict(
        success=e.success.copy(), collided=e.collided.copy(), route=e.route.copy(),
        side_changes=e.side_changes.copy(), finish_t=e.finish_t.copy(), event_noop=float(e.event_noop.mean()),
        seam_rms=float(np.sqrt(seam_sq / max(seam_n, 1))), du_rms=float(np.sqrt(du_sq / max(du_n, 1))))
