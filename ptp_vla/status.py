import glob, json, os, random, re
from collections import Counter, defaultdict
roots = [os.getcwd(), os.path.expanduser("~"), "/workspace", "/root", "/"]
def find(pat):
    for r in roots:
        for d in ("", "*/", "*/*/"):
            hit = glob.glob(os.path.join(r, d, pat))
            if hit:
                return hit[0]
R, L = find("ptp_vla/results"), find("ptp_vla/run.log") or find("ptp_run.log")
if not R:
    raise SystemExit("results folder not found: cd to the folder where you started the run")
def load(name):
    p = os.path.join(R, name)
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
quick = load("quick_cells.jsonl")
if quick or os.path.exists(os.path.join(R, "quick_results.json")):
    print(f"=== QUICK RUN: {len(quick)} of 80 jobs ({100 * len(quick) / 80:.0f}%)")
cells = load("benchmark_cells.jsonl")
dev = [c for c in cells if c["key"].startswith("dev|")]
test = [c for c in cells if c["key"].startswith("test|")]
s0 = len(load("stage0_cells.jsonl"))
done = s0 * 10 + len(dev) * 20 + len(test) * 50
print(f"=== PROGRESS: {done} of about 9500 episodes ({100 * done / 9500:.1f}%) | stage0 {s0}/15 | choosing {len(dev)}/280 | final test {len(test)}/60-80")
sec = [c["meta"].get("seconds") for c in dev + test if c["meta"].get("seconds")]
if sec:
    print(f"average job {sum(sec) / len(sec):.0f} s, peak VRAM {max(c['meta'].get('peak_vram_gb') or 0 for c in dev + test)} GB")

p = os.path.join(R, "stage0_results.json")
if os.path.exists(p):
    d = json.load(open(p))
    v, c = d["values"], json.load(open(os.path.join(R, "protocol_used.json")))["stage0"]["criteria"]
    print(f"\n=== STAGE 0: {d['decision']}")
    rows = [("G0 model can do the task", "competence", "G0_competence_min", "G0"), ("G1 noise changes the actions", "prefix_spread_p50", "G1_spread_min", "G1"),
            ("G2 noise acts locally in time", "locality", "G2_locality_min", "G2"), ("G3 carry keeps the mode", "keep_gain", "G3_keep_gain_min", "G3"), ("G4 carry smooths plans", "jitter_gain", "G4_jitter_gain_min", "G4")]
    for name, k, m, g in rows:
        x = v.get(k)
        print(f"  {'PASS' if d['gates'][g] else 'FAIL'}  {name:32s} value {'none' if x is None else round(x, 4)}  needed {c[m]}")
    print("  success in stage 0:", {k: round(x, 3) for k, x in d["success"].items()}, "episodes", d["episodes"])
    print(f"  states with two separate action modes: {d['multimodal_share']:.1%} of {d['states']}")
    print(f"  noise spread (action std units) p10/p50/p90: {d['spread']['prefix_p10']:.3f} / {d['spread']['prefix_p50']:.3f} / {d['spread']['prefix_p90']:.3f}")
    print(f"  noise locality within +-{d['locality']['band']} tokens: {d['locality']['within_band']:.3f} (random would be {d['locality']['null_uniform']:.3f})")
    print("  plan disagreement (overlap RMS):", {k: round(x, 4) for k, x in d["overlap_rms"].items()})
    print("  mode keep at same observation:", d["keep_same_observation"])

def table(group, title):
    by = defaultdict(dict)
    for c in group:
        k = c["key"].split("|")
        by[k[1]][tuple(k[2:])] = {r["episode"]: float(r["success"]) for r in c["rows"]}
    if not by:
        return
    print(f"\n=== {title} (success, and difference from the reference on the exact same jobs)")
    print("  Early numbers are noisy. Nothing here is the verdict.")
    ref = "native_L50" if "native_L50" in by else None
    rnd = random.Random(0)
    print(f"  {'method':16s} {'jobs':>4s} {'episodes':>8s} {'success':>8s}   vs {ref or 'none'} [95% interval on these episodes]")
    rate = lambda n: sum(sum(x.values()) for x in by[n].values()) / max(1, sum(len(x) for x in by[n].values()))
    for n in sorted(by, key=rate, reverse=True):
        count = sum(len(x) for x in by[n].values())
        line = f"  {n:16s} {len(by[n]):4d} {count:8d} {rate(n):8.3f}"
        if ref and n != ref:
            diffs = [by[n][k][e] - by[ref][k][e] for k in by[n] if k in by[ref] for e in by[n][k] if e in by[ref][k]]
            if diffs:
                boots = sorted(sum(rnd.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(1000))
                line += f"   {sum(diffs) / len(diffs) * 100:+6.1f} pts [{boots[25] * 100:+.1f}, {boots[974] * 100:+.1f}] on {len(diffs)} episodes"
        print(line)
    cond = defaultdict(lambda: defaultdict(list))
    for n, cs in by.items():
        for k, cell in cs.items():
            cond[k[0]][n] += list(cell.values())
    for name, ms in cond.items():
        print(f"  {name}: " + ", ".join(f"{n} {sum(x) / len(x):.2f}" for n, x in sorted(ms.items(), key=lambda t: -sum(t[1]) / len(t[1]))[:6]))
table(dev, "CHOOSING SETTINGS SO FAR")
table(test, "FINAL TEST SO FAR")
table(quick, "QUICK RUN SO FAR")
p = os.path.join(R, "quick_results.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("\n=== QUICK VERDICT:", d["verdict"], "(small run, large effects only)")
    print("success:", {k: round(v, 3) for k, v in d["success"].items()})
    for k in ("primary_same_minus_fresh", "shift_minus_fresh", "same_minus_native", "fresh_minus_native"):
        x = d[k]
        print(f"  {k}: {x['diff'] * 100:+.1f} pts, 95% interval {x['ci95'][0] * 100:+.1f} to {x['ci95'][1] * 100:+.1f}, these tasks only {x['ci95_these_tasks'][0] * 100:+.1f} to {x['ci95_these_tasks'][1] * 100:+.1f}")
p = os.path.join(R, "benchmark_results.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("\n=== FINAL VERDICT:", d["verdict"], "| PTP", d["ptp"], "| standard", d["standard"])
    print(json.dumps(d["primary_ptp_minus_standard"], indent=1))
if L:
    text = open(L, errors="replace").read().replace("\r", "\n")
    lines = [x for x in text.split("\n") if x.strip() and "Running rollout" not in x and "Stepping through" not in x]
    is_err = lambda x: re.match(r"^[A-Za-z_.]*(Error|Exception)\b", x)
    print(f"\n=== HEALTH: tracebacks {sum('Traceback (most recent' in x for x in lines)}, retries {sum('retrying ' in x for x in lines)}, worker restarts {sum('exited with code' in x for x in lines)}")
    for e, n in Counter(x[:110] for x in lines if is_err(x)).most_common(6):
        print(f"  {n}x {e}")
    for i, x in enumerate(lines):
        if "Traceback (most recent" in x:
            j = next((q for q in range(i + 1, min(i + 60, len(lines))) if is_err(lines[q])), i + 8)
            print("  first traceback ends with:")
            for y in lines[max(i, j - 5):j + 1]:
                print("   ", y[:170])
            break
    print("  latest:", lines[-1][:170] if lines else "")
