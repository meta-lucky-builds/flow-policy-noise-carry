"""Expert collection. Episodes split into segments at events, so no window crosses an event."""
import argparse
import multiprocessing as mp
import time
import numpy as np

from env import sample_scene, observe
from expert import plan

EVENTS = ("none", "nuisance", "goal", "obstacle", "velocity")
EVENT_P = (0.3, 0.1, 0.2, 0.2, 0.2)
TRAIN_BASE, VAL_BASE, TEST_BASE = 0, 1_000_000, 2_000_000  # test seeds are reserved, unused by Stage 0


def _route_for(x, goal, obst):
    return 0 if x[0] >= obst[0] - 0.05 or goal[0] <= obst[0] else None


def _try_plan(x, v, goal, obst, routes, Ns):
    for r in routes:
        for N in Ns:
            out = plan(x, v, goal, obst, r, N)
            if out is not None:
                return r, N, out
    return None


def episode(seed):
    rng = np.random.default_rng(seed)
    sc = sample_scene(rng)
    routes = list(rng.permutation([1, -1]))
    Ns = [int(n) for n in rng.permutation([80, 100, 120])]
    got = _try_plan(sc["start"], np.zeros(2), sc["goal"], sc["obst"], routes, Ns)
    stats = dict(first_certified=got is not None, event="none", post_certified=None, kept_route=None)
    if got is None:
        return None, stats
    route, N, (U, xs, vs) = got
    event = EVENTS[rng.choice(len(EVENTS), p=EVENT_P)]
    stats["event"] = event
    goal, obst, nuis = sc["goal"], sc["obst"], sc["nuis"]

    def seg(a, b, g, o, n, xs_, vs_, U_, r):
        T = b - a
        ob = observe(xs_[a:b], vs_[a:b], np.tile(g, (T, 1)), np.tile(o, (T, 1)), np.tile(n, (T, 1)))
        return dict(obs=ob, act=U_[a:b], route=r)

    if event == "none":
        return [seg(0, N, goal, obst, nuis, xs, vs, U, route)], stats

    te = int(rng.integers(N // 5, 3 * N // 5))
    if event == "nuisance":
        n2 = rng.normal(size=2)
        return [seg(0, te, goal, obst, nuis, xs, vs, U, route), seg(te, N, goal, obst, n2, xs, vs, U, route)], stats

    out = [seg(0, te, goal, obst, nuis, xs, vs, U, route)]
    x, v = xs[te], vs[te].copy()
    g2, o2 = goal.copy(), obst.copy()
    if event == "goal":
        a = rng.uniform(0, 2 * np.pi)
        g2 = g2 + rng.uniform(0.1, 0.3) * np.array([np.cos(a), np.sin(a)])
        g2 = np.clip(g2, [0.5, -0.6], [0.95, 0.6])
    elif event == "obstacle":
        # Pushes the obstacle toward the robot's current side, the case that can invalidate its route.
        side = route if route else 1
        o2 = o2 + np.array([rng.uniform(-0.05, 0.05), side * rng.uniform(0.1, 0.3)])
    else:
        a = rng.uniform(0, 2 * np.pi)
        v = np.clip(v + rng.uniform(0.1, 0.3) * np.array([np.cos(a), np.sin(a)]), -1, 1)
    rem = N - te
    Ns2 = [int(max(30, rem + d)) for d in rng.permutation([-10, 0, 20])]
    if x[0] >= o2[0] - 0.05:
        routes2 = [0]
    else:
        routes2 = [route, -route]  # retained intent first, the alternative only if it is not certified
    got2 = _try_plan(x, v, g2, o2, routes2, Ns2)
    stats["post_certified"] = got2 is not None
    if got2 is not None:
        r2, N2, (U2, xs2, vs2) = got2
        stats["kept_route"] = bool(r2 == route or r2 == 0)
        out.append(seg(0, N2, g2, o2, nuis, xs2, vs2, U2, r2 if r2 else route))
    return out, stats


def collect(base, n, pool, label):
    t0 = time.time()
    obs, act, ends, routes, stats = [], [], [], [], []
    for segs, st in pool.imap(episode, range(base, base + n), chunksize=8):
        stats.append(st)
        if segs is None:
            continue
        for s in segs:
            if len(s["act"]) == 0:
                continue
            obs.append(s["obs"]); act.append(s["act"]); routes.append(s["route"])
            ends.append(len(s["act"]))
    summary = dict(
        requested=n,
        certified=sum(s["first_certified"] for s in stats),
        post_attempted=sum(s["post_certified"] is not None for s in stats),
        post_certified=sum(bool(s["post_certified"]) for s in stats),
        kept_route=sum(bool(s["kept_route"]) for s in stats),
        seconds=round(time.time() - t0, 1),
        windows=int(sum(ends)),
    )
    print(label, summary, flush=True)
    return dict(obs=np.concatenate(obs).astype(np.float32), act=np.concatenate(act).astype(np.float32),
                seg_len=np.array(ends), seg_route=np.array(routes)), summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--out", default="data.npz")
    a = ap.parse_args()
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        tr, s1 = collect(TRAIN_BASE, a.train, pool, "train")
        va, s2 = collect(VAL_BASE, a.val, pool, "val")
    np.savez_compressed(a.out, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()})
