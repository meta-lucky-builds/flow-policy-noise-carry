"""FM training per spec 9.1 and 11.2. Writes a checkpoint at each validation point."""
import argparse
import copy
import json
import math
import time
import numpy as np
import torch

from model import FMPolicy


class Windows:
    """H-token windows that never cross a segment. Tail tokens repeat the last action and carry loss mask 0."""

    def __init__(self, obs, act, seg_len, H, lo=None, hi=None):
        self.obs, self.H = obs, H
        self.lo = act.min(0) if lo is None else lo
        self.hi = act.max(0) if hi is None else hi
        self.act = 2 * (act - self.lo) / np.maximum(self.hi - self.lo, 1e-6) - 1
        self.end = np.repeat(np.cumsum(seg_len), seg_len)
        self.n = len(obs)

    def batch(self, idx):
        pos = idx[:, None] + np.arange(self.H)
        mask = pos < self.end[idx][:, None]
        pos = np.minimum(pos, self.end[idx][:, None] - 1)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[pos]),
                torch.from_numpy(mask.astype(np.float32)))


def fm_loss(model, obs, a, mask, gen):
    z = torch.randn(a.shape, generator=gen, device=a.device)
    t = torch.rand(a.shape[0], generator=gen, device=a.device)
    x = (1 - t)[:, None, None] * z + t[:, None, None] * a
    v = model.velocity(x, t, obs)
    return (((v - (a - z)) ** 2).sum(-1) * mask).sum() / (2 * mask.sum()).clamp_min(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--updates", type=int, default=30000)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--time-only", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    d = np.load(a.data)
    tr = Windows(d["train_obs"], d["train_act"], d["train_seg_len"], 16)
    va = Windows(d["val_obs"], d["val_act"], d["val_seg_len"], 16, tr.lo, tr.hi)

    model = FMPolicy()
    model.obs_mean.copy_(torch.from_numpy(d["train_obs"].mean(0)))
    model.obs_std.copy_(torch.from_numpy(d["train_obs"].std(0) + 1e-6))
    model.act_lo.copy_(torch.from_numpy(tr.lo.copy()))
    model.act_hi.copy_(torch.from_numpy(tr.hi.copy()))
    model.to(dev)
    ema = copy.deepcopy(model)
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (nodecay if p.ndim < 2 or "norm" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    total = a.updates

    def lr_at(u):
        if u < 2000:
            return 3e-4 * (u + 1) / 2000
        f = (u - 2000) / max(total - 2000, 1)
        return 3e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))

    vidx = np.random.default_rng(12345).integers(0, va.n, 4096)
    t0 = time.time()
    log = []
    for u in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(u)
        obs, act, mask = (b.to(dev) for b in tr.batch(rng.integers(0, tr.n, 128)))
        loss = fm_loss(model, obs, act, mask, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if a.time_only and u + 1 == a.time_only:
            print(json.dumps(dict(updates_per_s=round(a.time_only / (time.time() - t0), 2))))
            raise SystemExit
        if (u + 1) % a.val_every == 0 or u + 1 == total:
            with torch.no_grad():
                g2 = torch.Generator(device=dev).manual_seed(999)
                vl = float(np.mean([fm_loss(ema, *(b.to(dev) for b in va.batch(vidx[i:i + 512])), g2).item()
                                    for i in range(0, 4096, 512)]))
            rec = dict(update=u + 1, train_loss=round(loss.item(), 5), val_loss=round(vl, 5), s=round(time.time() - t0))
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(dict(state={k: v.cpu() for k, v in ema.state_dict().items()}, version=f"seed{a.seed}-u{u + 1}", log=log), a.out)
