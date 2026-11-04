"""Stage 0 instruments and paired statistics. No LeRobot import, so they can be validated on known answers.

chunk_fn(z) must decode noise z of shape (B, H, D) at one fixed observation into chunks (B, H, A).
"""
import numpy as np
import torch

from carry import carry_source

ASHMAN_MIN = 3.5
MINOR_MIN = 0.1


def two_means_1d(p, iters=25):
    lo, hi = np.percentile(p, 10), np.percentile(p, 90)
    lab = p > (lo + hi) / 2
    for _ in range(iters):
        lab = np.abs(p - hi) < np.abs(p - lo)
        if lab.all() or (~lab).all():
            return lab, (lo + hi) / 2, 0.0, 0.0
        lo, hi = p[~lab].mean(), p[lab].mean()
    s = np.sqrt(p[lab].var() + p[~lab].var()) + 1e-12
    return lab, (lo + hi) / 2, float(np.sqrt(2) * abs(hi - lo) / s), float(min(lab.mean(), 1 - lab.mean()))


def spread(chunks, L):
    """Noise sensitivity: per-coordinate std over noise draws, averaged, for the executed prefix and whole chunk."""
    x = chunks.detach().cpu().numpy()
    return dict(prefix=float(x[:, :L].std(0).mean()), chunk=float(x.std(0).mean()))


def mode_screen(chunks):
    """Project sampled chunks on their first principal axis, then test for two separated modes."""
    f = chunks.detach().cpu().numpy().reshape(len(chunks), -1)
    mean = f.mean(0)
    c = f - mean
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    lab, thr, ash, minor = two_means_1d(c @ axis)
    return dict(mean=mean, axis=axis, threshold=thr, labels=lab, ashman=ash, minor=minor,
                multimodal=bool(ash > ASHMAN_MIN and minor >= MINOR_MIN))


def keep_same_observation(chunk_fn, z, screen, rhos, L, gen, modes=("shift", "same")):
    """Open-loop keep rate: carried noise decoded at the SAME observation, labelled on the original axis.

    The robot does not move between the two plans, so this isolates the noise effect and does not include
    the extra commitment that real state feedback would add.
    """
    side = lambda a: (a.detach().cpu().numpy().reshape(len(a), -1) - screen["mean"]) @ screen["axis"] > screen["threshold"]
    out = {}
    for m in modes:
        out[m] = {str(r): float((side(chunk_fn(carry_source(z, r, L, gen, m))) == screen["labels"]).mean()) for r in rhos}
    return out


def token_response(chunk_fn, H, D, gen, directions=2, delta=0.25, micro_batch=32):
    """M[i, j]: squared change of action token i when noise token j is perturbed. Finite difference, no gradients.

    A full Jacobian would need 1,600 backward passes per state for this checkpoint.
    """
    M = np.zeros((H, H))
    for _ in range(directions):
        base = torch.randn(1, H, D, generator=gen)
        z = base.repeat(H, 1, 1)
        for j in range(H):
            d = torch.randn(D, generator=gen)
            z[j, j] += delta * d / d.norm()
        a0 = chunk_fn(base).detach().cpu()
        parts = [chunk_fn(z[k:k + micro_batch]).detach().cpu() for k in range(0, H, micro_batch)]
        a = torch.cat(parts)
        M += ((a - a0) ** 2).mean(dim=2).numpy().T  # rows i (action token), columns j (noise token)
    return M / directions


def locality(M, band):
    i, j = np.indices(M.shape)
    near = np.abs(i - j) <= band
    return dict(within_band=float(M[near].sum() / max(M.sum(), 1e-30)), null_uniform=float(near.mean()), band=band)


def paired_difference(rows, a, b, nboot=4000, seed=0):
    """Mean success a-b on matched episodes, with two 95% intervals.

    ci95 bootstraps tasks and then episodes, so it speaks to tasks like these. ci95_these_tasks keeps the tasks
    fixed and bootstraps episodes only, so it speaks only to these tasks.
    rows: dicts with method, condition, suite, task, seed, episode, success. Unmatched episodes raise.
    """
    key = lambda r: (r["condition"], r["suite"], r["task"], r["seed"], r["episode"])
    A, B = {key(r): r["success"] for r in rows if r["method"] == a}, {key(r): r["success"] for r in rows if r["method"] == b}
    if not A or A.keys() != B.keys():
        raise ValueError(f"{a} and {b} are not fully paired: {len(A)} vs {len(B)} episodes")
    tasks = {}
    for k in A:
        tasks.setdefault((k[1], k[2]), []).append(float(A[k]) - float(B[k]))
    arrs = [np.array(v) for v in tasks.values()]
    rng = np.random.default_rng(seed)
    boots, fixed = [], []
    for _ in range(nboot):
        picks = rng.integers(0, len(arrs), len(arrs))
        boots.append(np.mean([rng.choice(arrs[t], len(arrs[t])).mean() for t in picks]))
        fixed.append(np.mean([rng.choice(x, len(x)).mean() for x in arrs]))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(diff=float(np.mean([x.mean() for x in arrs])), ci95=pct(boots), ci95_these_tasks=pct(fixed),
                n_episodes=len(A), n_tasks=len(arrs),
                discordant=[int(sum(A[k] and not B[k] for k in A)), int(sum(B[k] and not A[k] for k in A))])


def stage0_decision(g, crit):
    gates = dict(
        G0=g["competence"] >= crit["G0_competence_min"], G1=g["prefix_spread_p50"] >= crit["G1_spread_min"],
        G2=g["locality"] >= crit["G2_locality_min"], G3=g["keep_gain"] >= crit["G3_keep_gain_min"],
        G4=g["jitter_gain"] >= crit["G4_jitter_gain_min"])
    if not gates["G0"]:
        return "FIX-BASE", gates
    return ("GO" if gates["G1"] and gates["G2"] and (gates["G3"] or gates["G4"]) else "NO-GO"), gates


def benchmark_decision(primary, min_gain):
    """Verdict for the chosen carry setting against the best standard setting, on tasks unused for choosing."""
    if primary["ci95_these_tasks"][1] < 0:
        return "WORSE-THAN-STANDARD"
    if primary["diff"] >= min_gain and primary["ci95"][0] > 0:
        return "WIN"
    if primary["diff"] >= min_gain and primary["ci95_these_tasks"][0] > 0:
        return "WIN-ON-THESE-TASKS-ONLY"
    return "NO-IMPROVEMENT-SHOWN"


def choose(dev_scores, names, spec):
    """Highest dev success among names. Exact ties go to the lower rho, then the longer L."""
    return max(names, key=lambda n: (dev_scores[n], -spec[n]["rho"], spec[n]["L"]))
