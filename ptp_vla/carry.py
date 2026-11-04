"""K=1 PTP noise carry for a frozen LeRobot SmolVLA policy.

Patches one policy instance, so LeRobot's own rollout and eval_policy run unchanged:
  select_action       chooses the source noise when the action queue is empty
  _get_action_chunk   records plan-consistency metrics in normalized action units
  reset               clears all memory at the start of each rollout batch

Noise has shape (B, chunk_size, max_action_dim). For this checkpoint that is (B, 50, 32),
including the 25 padded dimensions the model still receives as noise.
"""
import torch

RHO_CAP = 0.95
MODES = ("native", "shift", "same", "cyc")


def carry_source(zprev, rho, L, gen, mode="shift"):
    """Z_j = rho zbar_j + sqrt(1-rho^2) eps_j.

    shift: aligned by L tokens, with a fresh tail. Memory ends after H/L replans.
    same:  same-index carry, not time-aligned. Memory decays as rho^m.
    cyc:   aligned by L tokens, with the executed tokens recycled into the tail. Keeps the overlap
           alignment of shift while memory decays as rho^m. The source stays N(0, I) and i.i.d. within a chunk.
    """
    if not 0.0 <= rho <= RHO_CAP:
        raise ValueError(f"rho must be in [0, {RHO_CAP}], got {rho}")
    B, H, D = zprev.shape
    if not 1 <= L <= H:
        raise ValueError(f"L must be in [1, {H}], got {L}")
    eps = torch.randn(zprev.shape, generator=gen, dtype=zprev.dtype)
    if mode == "shift":
        anchor = torch.zeros_like(zprev)
        anchor[:, : H - L] = zprev[:, L:]
        mask = torch.zeros(B, H, 1, dtype=zprev.dtype)
        mask[:, : H - L] = 1.0
    elif mode == "same":
        anchor, mask = zprev, torch.ones(B, H, 1, dtype=zprev.dtype)
    elif mode == "cyc":
        anchor, mask = torch.roll(zprev, -L, dims=1), torch.ones(B, H, 1, dtype=zprev.dtype)
    else:
        raise ValueError(f"unknown carry mode {mode}")
    r = rho * mask
    return r * anchor + torch.sqrt(1 - r**2) * eps


class ChunkMetrics:
    """Plan consistency between consecutive chunks. Sub-envs frozen after episode end are excluded."""

    def __init__(self):
        self.overlap_sq, self.overlap_n, self.seam_sq, self.seam_n = 0.0, 0, 0.0, 0
        self.du_sq, self.du_n, self.chunks = 0.0, 0, 0
        self._prev, self._prev_state = None, None

    def new_rollout(self):
        self._prev, self._prev_state = None, None

    def update(self, chunk, state, L):
        chunk, state = chunk.detach().float().cpu(), state.detach().float().cpu().reshape(len(chunk), -1)
        H = chunk.shape[1]
        self.chunks += 1
        if self._prev is None:
            active = torch.ones(len(chunk), dtype=torch.bool)
        else:
            active = (state != self._prev_state).any(dim=1)
            if active.any():
                seam = chunk[active, 0] - self._prev[active, L - 1]
                self.seam_sq += float((seam**2).mean(dim=1).sum())
                self.seam_n += int(active.sum())
                self.du_sq += float((seam**2).mean(dim=1).sum())
                self.du_n += int(active.sum())
                if H > L:
                    d = chunk[active, : H - L] - self._prev[active, L:]
                    self.overlap_sq += float((d**2).sum())
                    self.overlap_n += d.numel()
        if active.any() and L > 1:
            du = chunk[active, 1:L] - chunk[active, : L - 1]
            self.du_sq += float((du**2).mean(dim=2).sum())
            self.du_n += du.shape[0] * du.shape[1]
        self._prev, self._prev_state = chunk, state

    def summary(self):
        rms = lambda s, n: (s / n) ** 0.5 if n else None
        return dict(overlap_rms=rms(self.overlap_sq, self.overlap_n), seam_rms=rms(self.seam_sq, self.seam_n),
                    executed_du_rms=rms(self.du_sq, self.du_n), chunks=self.chunks,
                    units="normalized action (MEAN_STD), before unnormalizing")


class NoiseCarry:
    """mode='native' passes noise=None, leaving the checkpoint's own sampling path untouched."""

    def __init__(self, policy, rho=0.0, mode="shift", seed=0, on_reset=None, on_chunk=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "native" and rho != 0.0:
            raise ValueError("native mode has no carry")
        H, L = policy.config.chunk_size, policy.config.n_action_steps
        if mode in ("shift", "cyc") and rho > 0 and L >= H:
            raise ValueError(f"{mode} carry needs overlap, but n_action_steps={L} >= chunk_size={H}")
        self.policy, self.rho, self.mode = policy, rho, mode
        self.gen = torch.Generator("cpu").manual_seed(seed)
        self.on_reset, self.on_chunk = on_reset, on_chunk
        self.metrics = ChunkMetrics()
        self.resets = self.alignment_discards = self.proposals = 0
        self.last_noise = None
        self._z, self._pops = None, 0
        self._orig_select = policy.select_action
        self._orig_chunk = policy._get_action_chunk
        self._orig_reset = policy.reset
        policy.select_action = self._select_action
        policy._get_action_chunk = self._get_action_chunk
        policy.reset = self._reset

    def uninstall(self):
        for name in ("select_action", "_get_action_chunk", "reset"):
            del self.policy.__dict__[name]

    def _reset(self):
        self._orig_reset()
        self._z, self._pops = None, 0
        self.resets += 1
        self.metrics.new_rollout()
        if self.on_reset is not None:
            self.on_reset()

    def _next_noise(self, batch_size):
        cfg = self.policy.config
        shape = (batch_size, cfg.chunk_size, cfg.max_action_dim)
        L = cfg.n_action_steps
        valid = self._z is not None and self._z.shape == shape and self._pops == L
        if self._z is not None and not valid:
            self.alignment_discards += 1
        if valid:
            z = carry_source(self._z, self.rho, L, self.gen, self.mode)
        else:
            z = torch.randn(shape, generator=self.gen, dtype=torch.float32)
        self._z, self._pops = z, 0
        return z

    def _select_action(self, batch, *, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("NoiseCarry owns the noise; do not pass it to select_action")
        if self.policy._check_get_actions_condition() and self.mode != "native":
            state = batch["observation.state"]
            noise = self._next_noise(state.shape[0])
        action = self._orig_select(batch, noise=noise, **kwargs)
        self._pops += 1
        return action

    def _get_action_chunk(self, batch, *, noise=None, **kwargs):
        chunk = self._orig_chunk(batch, noise=noise, **kwargs)
        self.proposals += 1
        self.last_noise = noise
        self.metrics.update(chunk, batch["observation.state"], self.policy.config.n_action_steps)
        if self.on_chunk is not None:
            self.on_chunk(batch, chunk)
        return chunk
