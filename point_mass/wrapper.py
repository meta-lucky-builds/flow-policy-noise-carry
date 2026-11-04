"""K=1 PTP source: Z_j = rho_j zbar_j + sqrt(1-rho_j^2) eps_j, plus the execution ledger (spec 2, 5, 12)."""
import math
from dataclasses import dataclass, field
import numpy as np
import torch

RHO_CAP = 0.95


def carry_source(zprev, rho, L, gen, valid=None, mode="shift"):
    """Batched source draw.

    shift: spec default, aligned by consumed tokens with a fresh tail.
    same:  same-index carry, no time alignment.
    cyc:   aligned, with executed tokens recycled into the tail.
    valid: bool per batch element; False means reset, so rho=0 for that element.
    """
    if not 0.0 <= rho <= RHO_CAP:
        raise ValueError("rho outside [0, 0.95]")
    B, H, d = zprev.shape
    eps = torch.randn(zprev.shape, generator=gen, dtype=zprev.dtype)
    if mode == "shift":
        anchor = torch.zeros_like(zprev)
        anchor[:, :H - L] = zprev[:, L:]
        mask = torch.zeros(B, H, 1, dtype=zprev.dtype)
        mask[:, :H - L] = 1.0
    elif mode == "same":
        anchor, mask = zprev, torch.ones(B, H, 1, dtype=zprev.dtype)
    elif mode == "cyc":
        anchor, mask = torch.roll(zprev, -L, dims=1), torch.ones(B, H, 1, dtype=zprev.dtype)
    else:
        raise ValueError(mode)
    if valid is not None:
        mask = mask * valid.to(zprev.dtype)[:, None, None]
    r = rho * mask
    return r * anchor + torch.sqrt(1 - r ** 2) * eps


def latent_logprob(z, anchor, mask, rho):
    """Exact log density of section 14.3 for the K=1 tuple, with mask shape [H]."""
    r = rho * mask[:, None]
    var = 1 - r ** 2
    return float((-0.5 * ((z - r * anchor) ** 2 / var + torch.log(2 * math.pi * var))).sum())


def commitment_kl(anchor, mask, rho):
    """Conditional latent KL to N(0,I), per the D_j equation of section 5."""
    d = anchor.shape[-1]
    r2 = (rho * mask) ** 2
    return float((0.5 * (r2 * (anchor ** 2).sum(-1) + d * ((1 - r2) - 1 - torch.log(1 - r2)))).sum())


@dataclass
class Proposal:
    pid: int
    z: torch.Tensor
    version: str
    consumed: int = 0
    dispatched: dict = field(default_factory=dict)  # token index -> dispatch id


class LedgerError(RuntimeError):
    pass


class ExecutionLedger:
    """Maps proposal tokens to dispatch and acknowledgement. Ambiguity discards the anchor rather than guessing."""

    def __init__(self, version):
        self.version = version
        self.prop = None
        self._next_pid = 0
        self._next_did = 0
        self._dispatch = {}
        self._acked = set()
        self.next_ack = 0
        self.ambiguous = False
        self.reset_reason = None

    def register(self, z):
        self.prop = Proposal(self._next_pid, z.clone(), self.version)
        self._next_pid += 1
        self.ambiguous = False
        self.reset_reason = None
        self._dispatch = {}
        self._acked = set()
        self.next_ack = 0
        return self.prop.pid

    def dispatch(self, pid, idx):
        p = self.prop
        if p is None or pid != p.pid:
            raise LedgerError("dispatch from a stale proposal")
        if idx in p.dispatched:
            raise LedgerError("token would execute twice")
        if idx != len(p.dispatched):
            self._flag("skipped_token")
        did = self._next_did
        self._next_did += 1
        p.dispatched[idx] = did
        self._dispatch[did] = idx
        return did

    def ack(self, did, status="ok"):
        if did in self._acked:
            self._flag("duplicate_ack")
            return
        if did not in self._dispatch:
            self._flag("unknown_ack")
            return
        self._acked.add(did)
        idx = self._dispatch[did]
        if idx != self.next_ack:
            self._flag("out_of_order_ack")
        self.next_ack = idx + 1
        if status != "ok":
            self._flag(f"token_{status}")
        elif not self.ambiguous:
            self.prop.consumed = self.next_ack

    def _flag(self, why):
        self.ambiguous = True
        self.reset_reason = self.reset_reason or why

    def anchor(self, current_version=None):
        """Returns (zbar, mask), both zero when alignment is not trustworthy."""
        p = self.prop
        if p is None:
            return None, None
        H = p.z.shape[0]
        zbar, mask = torch.zeros_like(p.z), torch.zeros(H, dtype=p.z.dtype)
        if (current_version or self.version) != p.version:
            self.reset_reason = "version_change"
            return zbar, mask
        if len(self._acked) != len(self._dispatch):
            self._flag("unacknowledged_tokens")
        if self.ambiguous:
            return zbar, mask
        c = p.consumed
        if c < H:
            zbar[:H - c] = p.z[c:]
            mask[:H - c] = 1.0
        return zbar, mask


def propose(policy, obs, ledger, rho, gen, steps=10):
    """Single-environment reference path of section 12. Returns (pid, actions, logp)."""
    H, d = policy.H, policy.act_dim
    z_shape = (H, d)
    eps = torch.randn(z_shape, generator=gen)
    zbar, mask = ledger.anchor(policy.version)
    if zbar is None:
        zbar, mask = torch.zeros(z_shape), torch.zeros(H)
    r = (rho * mask)[:, None]
    z = r * zbar + torch.sqrt(1 - r ** 2) * eps
    pid = ledger.register(z)
    actions = policy.decode(z[None], torch.as_tensor(np.asarray(obs), dtype=torch.float32)[None], steps)[0]
    return pid, actions, latent_logprob(z, zbar, mask, rho)
