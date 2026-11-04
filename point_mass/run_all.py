"""Full pipeline: train, Stage 0, selection on validation scenes, then one locked test. Resumes from outdir."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

import stage0
from bench import CONDITIONS, rollout
from gen_data import TEST_BASE, VAL_BASE


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def train_all(a, seeds):
    speed = None
    for s in seeds:
        ck = os.path.join(a.outdir, f"ckpt{s}.pt")
        if os.path.exists(ck) and os.path.exists(ck + ".done"):
            continue
        if speed is None:
            out = subprocess.run([sys.executable, "train.py", "--data", a.data, "--device", a.device, "--updates", "300",
                                  "--time-only", "200", "--out", "/tmp/t.pt"], capture_output=True, text=True, check=True)
            speed = json.loads(out.stdout.strip().splitlines()[-1])["updates_per_s"]
            updates = a.updates or int(np.clip(speed * a.train_seconds, 30000, 100000)) // 1000 * 1000
            log("speed", speed, "updates per seed", updates)
        subprocess.run([sys.executable, "-u", "train.py", "--data", a.data, "--device", a.device, "--updates", str(updates),
                        "--seed", str(s), "--out", ck], check=True)
        open(ck + ".done", "w").write("ok")


def load(path, device):
    m = stage0.load(path).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_cell(store_path, store, key, fn):
    """Each finished rollout is persisted, so a Colab disconnect only loses the one in progress."""
    if key in store:
        return store[key]
    t = time.time()
    r = fn()
    store[key] = {k: (v.astype(int).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    store[key]["seconds"] = round(time.time() - t, 1)
    save(store_path, store)
    return store[key]


def select(val, cands, conds, seeds):
    """Protocol rule: lowest rho within one paired SE of the mode's best, or 0 if fresh is within it."""
    per_scene = lambda name: np.mean([[val[f"{s}|{c}|{name}"]["success"] for c in conds] for s in seeds], axis=(0, 1))
    fresh = per_scene("shift_0.0")
    out = {}
    for mode, rhos in cands.items():
        scores = {r: per_scene(f"{mode}_{r}") for r in rhos}
        best = max(rhos, key=lambda r: scores[r].mean())
        within = lambda x: (scores[best] - x).mean() <= (scores[best] - x).std(ddof=1) / np.sqrt(len(x))
        pick = 0.0 if within(fresh) else next(r for r in sorted(rhos) if within(scores[r]))
        out[mode] = dict(rho=pick, best_rho=best, best_score=float(scores[best].mean()), fresh_score=float(fresh.mean()),
                         scores={str(r): float(v.mean()) for r, v in scores.items()})
    return out


def analyze(test, methods, conds, seeds, nboot=2000):
    rng = np.random.default_rng(0)
    table = {}
    for name in methods:
        cells = [[test[f"{s}|{c}|{name}"] for c in conds] for s in seeds]
        table[name] = dict(
            success=float(np.mean([[np.mean(x["success"]) for x in r] for r in cells])),
            by_condition={c: float(np.mean([np.mean(test[f"{s}|{c}|{name}"]["success"]) for s in seeds])) for c in conds},
            by_seed={str(s): float(np.mean([np.mean(test[f"{s}|{c}|{name}"]["success"]) for c in conds])) for s in seeds},
            collision=float(np.mean([[np.mean(x["collided"]) for x in r] for r in cells])),
            side_changes=float(np.mean([[np.mean(x["side_changes"]) for x in r] for r in cells])),
            seam_rms=float(np.mean([[x["seam_rms"] for x in r] for r in cells])),
            du_rms=float(np.mean([[x["du_rms"] for x in r] for r in cells])))
    diffs = {}
    base = "fresh_L4"
    for name in methods:
        if name == base:
            continue
        d = np.array([[np.array(test[f"{s}|{c}|{name}"]["success"], float) - np.array(test[f"{s}|{c}|{base}"]["success"], float)
                       for c in conds] for s in seeds])  # seeds, conds, scenes
        n = d.shape[2]
        boots = [d[:, :, rng.integers(0, n, n)].mean() for _ in range(nboot)]
        diffs[name] = dict(diff=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           by_seed=[float(v) for v in d.mean((1, 2))], by_condition={c: float(d[:, i].mean()) for i, c in enumerate(conds)})
    return table, diffs


def verdict(sel, diffs):
    if sel["shift"]["rho"] == 0.0:
        return "NULL: selection returned rho=0 for shift carry"
    p = diffs["shift_sel"]
    ok = p["ci95"][0] > 0 and p["diff"] >= 0.03 and (all(v > 0 for v in p["by_seed"]))
    return "WIN" if ok else "NO-WIN under the declared rule"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-seconds", type=float, default=1500)
    ap.add_argument("--updates", type=int, default=0)
    ap.add_argument("--val-scenes", type=int, default=1000)
    ap.add_argument("--test-scenes", type=int, default=2000)
    ap.add_argument("--stage0-episodes", type=int, default=500)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    proto = json.load(open("bench_protocol.json"))
    shutil.copy("bench_protocol.json", a.outdir)
    shutil.copy("stage0_criteria.json", a.outdir)
    seeds = list(range(a.seeds))
    if not a.skip_train:
        train_all(a, seeds)
    log("training done")

    s0 = os.path.join(a.outdir, "stage0_results.json")
    if not os.path.exists(s0):
        subprocess.run([sys.executable, "-u", "stage0.py", "--device", a.device, "--ckpt", os.path.join(a.outdir, "ckpt0.pt"),
                        "--out", s0, "--episodes", str(a.stage0_episodes), "--diag-episodes", str(a.stage0_episodes)], check=True)
    log("stage0", json.load(open(s0))["decision"])

    pol = {s: load(os.path.join(a.outdir, f"ckpt{s}.pt"), a.device) for s in seeds}
    cands = proto["selection"]["candidates"]
    vp, val = os.path.join(a.outdir, "val_cells.json"), {}
    if os.path.exists(vp):
        val = json.load(open(vp))
    vs = list(range(VAL_BASE, VAL_BASE + a.val_scenes))
    cfgs = [("shift", 0.0)] + [(m, r) for m, rs in cands.items() for r in rs]
    for s in seeds:
        for c in CONDITIONS:
            for m, r in cfgs:
                run_cell(vp, val, f"{s}|{c}|{m}_{r}",
                         lambda: rollout(pol[s], vs, c, 4, r, m, gen_seed=11 + s))
        log("validation done for training seed", s)
    sel = select(val, cands, CONDITIONS, seeds)
    save(os.path.join(a.outdir, "selection.json"), sel)
    log("selected", {k: v["rho"] for k, v in sel.items()})

    methods = {
        "fresh_L4": dict(L=4, rho=0.0), "shift_sel": dict(L=4, rho=sel["shift"]["rho"], mode="shift"),
        "same_sel": dict(L=4, rho=sel["same"]["rho"], mode="same"), "cyc_sel": dict(L=4, rho=sel["cyc"]["rho"], mode="cyc"),
        "fresh_L2": dict(L=2, rho=0.0), "fresh_L8": dict(L=8, rho=0.0),
        "shift_sel_L8": dict(L=8, rho=sel["shift"]["rho"], mode="shift"), "ensemble": dict(L=1, ensemble=True)}
    tp, test = os.path.join(a.outdir, "test_cells.json"), {}
    if os.path.exists(tp):
        test = json.load(open(tp))
    ts = list(range(TEST_BASE, TEST_BASE + a.test_scenes))
    for s in seeds:
        for c in CONDITIONS:
            for name, kw in methods.items():
                run_cell(tp, test, f"{s}|{c}|{name}", lambda: rollout(pol[s], ts, c, gen_seed=101 + s, **kw))
        log("test done for training seed", s)
    table, diffs = analyze(test, list(methods), CONDITIONS, seeds)
    v = verdict(sel, diffs)
    save(os.path.join(a.outdir, "benchmark_summary.json"),
         dict(verdict=v, selection={k: x["rho"] for k, x in sel.items()}, methods=table, vs_fresh_L4=diffs))
    log("VERDICT", v)
    print(json.dumps(dict(verdict=v, success={k: round(x["success"], 3) for k, x in table.items()},
                          diff_vs_fresh={k: [round(x["diff"], 3), [round(c, 3) for c in x["ci95"]]] for k, x in diffs.items()}), indent=1))
