"""Stage 0 screen: can noise carry act on this checkpoint at all? Uses validation seeds only, no tuning."""
import argparse
import json
import numpy as np
import torch
from torch.func import jacrev, vmap

from env import BatchEnv, OBST_R, step
from gen_data import VAL_BASE
from model import FMPolicy
from wrapper import carry_source

H, D, L, STEPS = 16, 2, 4, 10
RHOS = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
MODES = ["shift", "same", "cyc"]


def _dev(policy):
    return getattr(policy, "act_lo", torch.zeros(1)).device


def solve(policy, z, obs):
    """Runs the solver on the policy's device. Noise and all bookkeeping stay on the CPU."""
    return policy.solve(z.to(_dev(policy)), obs.to(_dev(policy)), STEPS).cpu()


def adapt(policy, x):
    return policy.adapter(x.to(_dev(policy))).cpu()


def load(path):
    c = torch.load(path, weights_only=False, map_location="cpu")
    m = FMPolicy(**c.get("cfg", {}))
    m.load_state_dict(c["state"])
    m.version = c["version"]
    return m.eval()


@torch.no_grad()
def rollout(policy, seeds, rho=0.0, mode="shift", gen_seed=0, record=False):
    """Closed-loop, same scheduler and solver for every setting; only the source law differs."""
    gen = torch.Generator().manual_seed(gen_seed)
    e = BatchEnv(seeds)
    n = len(seeds)
    z = torch.randn(n, H, D, generator=gen)
    visited, seam, jerk_sum, jerk_n = [], [], 0.0, 0
    first, prev_act, prev_u = True, None, None
    while not e.done.all() and e.t < 200:
        if not first:
            z = carry_source(z, rho, L, gen, mode=mode)
        act = adapt(policy, solve(policy, z, torch.from_numpy(e.obs()).float())).numpy()
        active = ~e.done
        if record:
            visited.append(e.obs()[active].copy())
        if prev_act is not None:
            seam.append(np.linalg.norm(prev_act[active, L - 1] - act[active, 0], axis=1))
        for j in range(L):
            act_now = ~e.done
            if prev_u is not None:
                du = act[:, j] - prev_u
                jerk_sum += float((du[act_now] ** 2).sum())
                jerk_n += int(act_now.sum())
            prev_u = act[:, j]
            e.step(act[:, j])
        prev_act, first = act, False
    return dict(
        success=e.success, collided=e.collided, route=e.route, side_changes=e.side_changes, finish_t=e.finish_t,
        seam_rms=float(np.sqrt(np.mean(np.concatenate(seam) ** 2))) if seam else 0.0,
        du_rms=float(np.sqrt(jerk_sum / max(jerk_n, 1))),
        visited=np.concatenate(visited) if record else None)


def two_means_1d(p, iters=25):
    """Returns labels, centers, Ashman D and minority mass for a 1-D projection."""
    lo, hi = np.percentile(p, 10), np.percentile(p, 90)
    for _ in range(iters):
        lab = np.abs(p - hi) < np.abs(p - lo)
        if lab.all() or (~lab).all():
            return lab, (lo, hi), 0.0, 0.0
        lo, hi = p[~lab].mean(), p[lab].mean()
    s = np.sqrt(p[lab].var() + p[~lab].var()) + 1e-9
    return lab, (lo, hi), float(np.sqrt(2) * abs(hi - lo) / s), float(min(lab.mean(), 1 - lab.mean()))


@torch.no_grad()
def sample_chunks(policy, ob, M, gen):
    z = torch.randn(M, H, D, generator=gen)
    x = solve(policy, z, torch.as_tensor(ob, dtype=torch.float32).expand(M, -1))
    return z, x


def screen_state(policy, ob, M, gen):
    z, x = sample_chunks(policy, ob, M, gen)
    f = x.reshape(M, -1).numpy()
    c = f - f.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    p = c @ vt[0]
    lab, centers, ash, minor = two_means_1d(p)
    return dict(z=z, x=x, lab=lab, ashman=ash, minor=minor, spread=float(np.sqrt((c ** 2).sum(1).mean() / (H * D))),
                prefix_spread=float(x[:, :L].std(0).mean()))


def jacobian_mass(policy, obs_batch, gen):
    """Mean squared Jacobian d A_i / d z_j, summed over action dims, per token pair."""
    def f(z, ob):
        return policy.solve(z[None], ob[None], STEPS)[0]
    out = []
    for ob in obs_batch:
        z = torch.randn(H, D, generator=gen).to(_dev(policy))
        J = jacrev(f)(z, torch.as_tensor(ob, dtype=torch.float32).to(z.device)).detach()  # H,D,H,D
        out.append((J ** 2).sum((1, 3)).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def keep_rates(policy, states, M, gen, rhos, modes):
    """Keep = new plan's overlap sits nearer the old plan's own cluster, after really executing L steps."""
    res = {m: {r: [] for r in rhos} for m in modes}
    for s in states:
        ob, st = s["ob"], s["screen"]
        z, x, lab = st["z"], st["x"], st["lab"]
        tail_a, tail_b = x[~lab][:, L:].mean(0), x[lab][:, L:].mean(0)
        u = adapt(policy, x)[:, :L].numpy()
        # L steps of the true dynamics from this state, one rollout per sampled chunk.
        p, v = np.tile(ob[:2], (M, 1)), np.tile(ob[2:4], (M, 1))
        for j in range(L):
            p, v = step(p, v, u[:, j])
        nob = np.tile(ob, (M, 1)); nob[:, :2] = p; nob[:, 2:4] = v
        nobt = torch.from_numpy(nob).float()
        for mode in modes:
            for r in rhos:
                z2 = carry_source(z, r, L, gen, mode=mode)
                head = solve(policy, z2, nobt)[:, :H - L]
                to_b = ((head - tail_b) ** 2).sum((1, 2))
                to_a = ((head - tail_a) ** 2).sum((1, 2))
                res[mode][r].append(float(((to_b < to_a).numpy() == lab).mean()))
    return {m: {str(r): float(np.mean(v)) for r, v in d.items()} for m, d in res.items()}


@torch.no_grad()
def jitter(policy, obs_batch, M, gen, rhos, modes):
    """Overlap RMS between old plan tail and new plan head, after executing L steps."""
    res = {m: {r: [] for r in rhos} for m in modes}
    for ob in obs_batch:
        z = torch.randn(M, H, D, generator=gen)
        obt = torch.as_tensor(ob, dtype=torch.float32).expand(M, -1)
        x = solve(policy, z, obt)
        u = adapt(policy, x)[:, :L].numpy()
        p, v = np.tile(ob[:2], (M, 1)), np.tile(ob[2:4], (M, 1))
        for j in range(L):
            p, v = step(p, v, u[:, j])
        nob = np.tile(ob, (M, 1)); nob[:, :2] = p; nob[:, 2:4] = v
        nobt = torch.from_numpy(nob).float()
        for mode in modes:
            for r in rhos:
                n = solve(policy, carry_source(z, r, L, gen, mode=mode), nobt)
                res[mode][r].append(float(((n[:, :H - L] - x[:, L:]) ** 2).mean()))
    return {m: {str(r): float(np.sqrt(np.mean(v))) for r, v in d.items()} for m, d in res.items()}


def wilson(k, n):
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--diag-episodes", type=int, default=500)
    ap.add_argument("--out", default="stage0_results.json")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    crit = json.load(open("stage0_criteria.json"))
    policy = load(a.ckpt).to(a.device)
    for p in policy.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(2026)
    rs = np.random.default_rng(2026)
    seeds = list(range(VAL_BASE, VAL_BASE + a.episodes))

    base = rollout(policy, seeds, 0.0, record=True, gen_seed=1)
    succ = int(base["success"].sum())
    print("fresh success", succ, "/", a.episodes, wilson(succ, a.episodes), flush=True)
    vis = base["visited"]
    approach = vis[(vis[:, 0] < vis[:, 6] - OBST_R) & (vis[:, 0] > -0.75)]
    pick = rs.choice(len(approach), min(a.states, len(approach)), replace=False)
    pick_all = rs.choice(len(vis), min(a.states, len(vis)), replace=False)

    M = 128
    screened = []
    for ob in approach[pick]:
        screened.append(dict(ob=ob, screen=screen_state(policy, ob, M, gen)))
    ash = np.array([s["screen"]["ashman"] for s in screened])
    minor = np.array([s["screen"]["minor"] for s in screened])
    multi = (ash > crit["G1_multimodal"]["ashman_min"]) & (minor >= 0.1)
    print("multimodal share", multi.mean(), flush=True)
    all_scr = [screen_state(policy, ob, M, gen) for ob in vis[pick_all[:200]]]
    act_scale = float(((policy.act_hi - policy.act_lo) / 2).mean())

    J = jacobian_mass(policy, vis[pick_all[:64]], gen)
    Jm = J.mean(0)
    i, j = np.indices((H, H))
    within3 = float(Jm[np.abs(i - j) <= 3].sum() / Jm.sum())
    null_within3 = float((np.abs(i - j) <= 3).mean())
    a1, a2 = Jm[:H - L, :H - L].ravel(), Jm[L:, L:].ravel()
    shift_corr = float(np.corrcoef(a1, a2)[0, 1])
    print("locality", within3, "null", null_within3, "shift corr", shift_corr, flush=True)

    multi_states = [s for s, m in zip(screened, multi) if m][:60]
    keep = keep_rates(policy, multi_states, M, gen, RHOS, MODES) if multi_states else None
    print("keep", json.dumps(keep), flush=True)
    jit = jitter(policy, vis[pick_all[:100]], 64, gen, RHOS, MODES)
    print("jitter", json.dumps(jit), flush=True)

    diag = {}
    dseeds = list(range(VAL_BASE, VAL_BASE + a.diag_episodes))
    for mode in MODES:
        for r in RHOS:
            if r == 0 and mode != "shift":
                continue
            o = rollout(policy, dseeds, r, mode, gen_seed=100)
            k = int(o["success"].sum())
            diag[f"{mode}/{r}"] = dict(
                success=round(k / a.diag_episodes, 3), ci95=wilson(k, a.diag_episodes),
                collision=round(float(o["collided"].mean()), 3), route_up=round(float((o["route"] > 0).mean()), 3),
                route_down=round(float((o["route"] < 0).mean()), 3), side_changes=round(float(o["side_changes"].mean()), 3),
                seam_rms=round(o["seam_rms"], 4), du_rms=round(o["du_rms"], 4),
                finish_t=round(float(o["finish_t"][o["success"]].mean()), 1) if k else None)
            print(mode, r, diag[f"{mode}/{r}"], flush=True)

    g0 = succ / a.episodes
    g1 = float(multi.mean())
    g3 = (keep["shift"]["0.95"] - keep["shift"]["0.0"]) if keep else None
    g4 = 1 - jit["shift"]["0.85"] / jit["shift"]["0.0"]
    gates = dict(
        G0=dict(value=g0, passed=g0 >= crit["G0_competence"]["min"]),
        G1=dict(value=g1, passed=g1 >= crit["G1_multimodal"]["min"]),
        G2=dict(value=within3, passed=within3 >= crit["G2_locality"]["min"]),
        G3=dict(value=g3, passed=g3 is not None and g3 >= crit["G3_mode_gain"]["min"]),
        G4=dict(value=g4, passed=g4 >= crit["G4_jitter_gain"]["min"]))
    if not gates["G0"]["passed"]:
        decision = "FIX-BASE"
    elif gates["G1"]["passed"] and gates["G2"]["passed"] and (gates["G3"]["passed"] or gates["G4"]["passed"]):
        decision = "GO"
    else:
        decision = "NO-GO"
    out = dict(
        version=policy.version, decision=decision, gates=gates,
        fresh=dict(success=g0, ci95=wilson(succ, a.episodes), collision=float(base["collided"].mean()),
                   route_up=float((base["route"] > 0).mean()), route_down=float((base["route"] < 0).mean())),
        sensitivity=dict(
            approach_prefix_spread_p50=float(np.median([s["screen"]["prefix_spread"] for s in screened])),
            approach_prefix_spread_p90=float(np.percentile([s["screen"]["prefix_spread"] for s in screened], 90)),
            all_chunk_spread_p50=float(np.median([s["spread"] for s in all_scr])),
            all_chunk_spread_p90=float(np.percentile([s["spread"] for s in all_scr], 90)),
            units="normalized action; full range is [-1,1]"),
        multimodal=dict(share=g1, n_states=len(screened), ashman_p50=float(np.median(ash)), n_used_for_keep=len(multi_states)),
        locality=dict(within3=within3, null_within3=null_within3, shift_equivariance_corr=shift_corr,
                      row8=[round(float(v), 4) for v in (Jm[8] / Jm[8].sum())]),
        keep=keep, jitter=jit, closed_loop_diagnostic=diag)
    json.dump(out, open(a.out, "w"), indent=1)
    np.save("stage0_jacobian.npy", Jm)
    print("DECISION", decision, json.dumps(gates), flush=True)
