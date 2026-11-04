"""Recomputes the benchmark statistics from raw episodes and collects the stored Stage 0 values.

    python analysis/analyze_results.py

Writes results/smolvla_libero/summary.json and docs/figures/effects.png.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "smolvla_libero"
METHODS = ("native_L50", "fresh_L10", "shift_L10_r90", "same_L10_r90")
CONDITIONS = ("clean", "delay4")
COMPARISONS = (("fresh_L10", "native_L50"), ("shift_L10_r90", "fresh_L10"), ("same_L10_r90", "fresh_L10"))


def load_cells():
    """Maps (method, condition, suite, task) to the job's episode rows."""
    cells = {}
    for line in (RESULTS / "quick_cells.jsonl").read_text().splitlines():
        cell = json.loads(line)
        _, method, condition, suite, task = cell["key"].split("|")
        cells[(method, condition, suite, int(task))] = cell
    return cells


def tasks_of(cells):
    return sorted({k[2:] for k in cells})


def success(cells, method, condition, task):
    return np.array([r["success"] for r in cells[(method, condition) + task]["rows"]], dtype=float)


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n) if n else 1.0


def compare(cells, a, b, conditions, nboot=4000, seed=0):
    """Paired difference a-b. One interval resamples tasks and episodes, the other keeps the tasks fixed."""
    per_task = [np.concatenate([success(cells, a, c, t) - success(cells, b, c, t) for c in conditions]) for t in tasks_of(cells)]
    rng = np.random.default_rng(seed)
    resampled, fixed = [], []
    for _ in range(nboot):
        picks = rng.integers(0, len(per_task), len(per_task))
        resampled.append(np.mean([rng.choice(per_task[t], len(per_task[t])).mean() for t in picks]))
        fixed.append(np.mean([rng.choice(x, len(x)).mean() for x in per_task]))
    d = np.concatenate(per_task)
    a_only, b_only = int((d > 0).sum()), int((d < 0).sum())
    return dict(
        diff_pts=round(float(d.mean()) * 100, 2),
        ci95_tasks_and_episodes_pts=[round(float(x) * 100, 1) for x in np.percentile(resampled, [2.5, 97.5])],
        ci95_these_tasks_pts=[round(float(x) * 100, 1) for x in np.percentile(fixed, [2.5, 97.5])],
        a_only=a_only, b_only=b_only, mcnemar_p=float(f"{exact_mcnemar(a_only, b_only):.3g}"),
        tasks_up=int(sum(x.mean() > 0 for x in per_task)), tasks_down=int(sum(x.mean() < 0 for x in per_task)), n_episodes=len(d))


def summarize():
    cells = load_cells()
    tasks = tasks_of(cells)
    rate = lambda m, conds, ts=tasks: float(np.mean([success(cells, m, c, t).mean() for c in conds for t in ts]))
    by_suite = lambda s: [t for t in tasks if t[0] == s]
    hashes = {k: [r["start_hash"] for r in c["rows"]] for k, c in cells.items()}
    mismatches = sum(a != b for k, h in hashes.items() for a, b in zip(h, hashes[("native_L50",) + k[1:]]))
    stage0 = json.loads((RESULTS / "stage0_results.json").read_text())
    token_mass = np.array(stage0["token_response"])
    return dict(
        integrity=dict(jobs=len(cells), episodes=sum(len(c["rows"]) for c in cells.values()), start_state_mismatches=mismatches,
                       peak_model_vram_gb=max(c["meta"]["peak_vram_gb"] for c in cells.values())),
        success_pct={m: dict(overall=round(rate(m, CONDITIONS) * 100, 2), clean=round(rate(m, ("clean",)) * 100, 2), delay4=round(rate(m, ("delay4",)) * 100, 2),
                             **{s: round(rate(m, CONDITIONS, by_suite(s)) * 100, 2) for s in ("libero_10", "libero_goal")}) for m in METHODS},
        effects={f"{a} - {b}": {name: compare(cells, a, b, conds) for name, conds in (("overall", CONDITIONS), ("clean", ("clean",)), ("delay4", ("delay4",)))} for a, b in COMPARISONS},
        shift_minus_same=compare(cells, "shift_L10_r90", "same_L10_r90", CONDITIONS),
        plan_consistency={m: {c: {f: (None if cells[(m, c) + tasks[0]]["meta"][f] is None else round(float(np.sqrt(np.mean([cells[(m, c) + t]["meta"][f] ** 2 for t in tasks]))), 3)) for f in ("overlap_rms", "seam_rms", "executed_du_rms")} for c in CONDITIONS} for m in METHODS},
        stage0=dict(decision=stage0["decision"], values=stage0["values"], gates=stage0["gates"], success=stage0["success"], multimodal_share=stage0["multimodal_share"],
                    keep_same_observation=stage0["keep_same_observation"], overlap_rms=stage0["overlap_rms"], spread=stage0["spread"],
                    share_of_response_in_top_two_action_tokens=round(float(np.sort(token_mass.sum(1))[-2:].sum()), 3)))


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, mark = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    rows = [("Replan every 10 steps vs every 50", "fresh_L10 - native_L50"), ("Shift carry vs fresh noise", "shift_L10_r90 - fresh_L10"), ("Same-index carry vs fresh noise", "same_L10_r90 - fresh_L10")]
    labels = {"clean": "clean", "delay4": "0.2 s delay"}
    entries = [(f"{title}, {labels[c]}", summary["effects"][key][c]) for title, key in rows for c in ("clean", "delay4")]
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200, facecolor=surface)
    ax.set_facecolor(surface)
    for i, (_, e) in enumerate(entries):
        y = len(entries) - 1 - i
        low, high = e["ci95_tasks_and_episodes_pts"]
        ax.plot([low, high], [y, y], color=mark, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter([e["diff_pts"]], [y], s=64, color=mark, edgecolor=surface, linewidth=2, zorder=4)
        ax.text(max(high, 0) + 1.2, y, f"{e['diff_pts']:+.1f}", va="center", ha="left", color=ink, fontsize=10)
    ax.axvline(0, color=muted, linewidth=1, zorder=2)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([name for name, _ in reversed(entries)], color=ink, fontsize=10)
    ax.set_xlabel("Change in task success, percentage points (dot) with 95% interval (line)", color=muted, fontsize=9.5)
    ax.tick_params(axis="x", colors=muted, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=grid, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-34, 42)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    result = summarize()
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    plot(result, ROOT / "docs" / "figures" / "effects.png")
    print(json.dumps(dict(success=result["success_pct"], effects=result["effects"], shift_minus_same=result["shift_minus_same"], integrity=result["integrity"]), indent=1))
