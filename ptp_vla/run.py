"""SmolVLA on LIBERO: does PTP noise carry improve on standard flow matching?

  python run.py all   runs verify, stage0 and benchmark in order. Rerunning skips finished work.

Each cell gets freshly built environments: LiberoEnv advances init_state_id on every reset, so reusing
environments across methods would silently give each method different starting states.
"""
import os

# Each simulator is its own process, so thread pools inside them would only fight over cores.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import platform
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    import fcntl
except ImportError:
    fcntl = None

from carry import NoiseCarry, carry_source
import metrics
import parallel

HERE = Path(__file__).parent


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def make_env_cfg(proto, suite, task_ids):
    from lerobot.envs.configs import LiberoEnv

    e = proto["env"]
    return LiberoEnv(task=suite, task_ids=task_ids, control_mode=e["control_mode"], init_states=e["init_states"],
                     hard_reset=e["hard_reset"], observation_height=e["observation_size"], observation_width=e["observation_size"])


def build(proto, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pin, envp = proto["pins"], proto["env"]
    cfg = PreTrainedConfig.from_pretrained(pin["checkpoint"], revision=pin["checkpoint_revision"])
    cfg.pretrained_path = Path(pin["checkpoint"])
    cfg.pretrained_revision = pin["checkpoint_revision"]
    cfg.device = device
    policy = make_policy(cfg=cfg, env_cfg=make_env_cfg(proto, "libero_10", [0]), rename_map=envp["rename_map"])
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=cfg.pretrained_path, pretrained_revision=cfg.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": envp["rename_map"]}})
    return cfg, policy, pre, post


class EnvPre:
    """Wraps the env preprocessor. Records each rollout's starting states and optionally serves stale observations."""

    def __init__(self, inner, delay=0):
        self.inner, self.delay = inner, delay
        self.start_hashes = []
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=self.delay + 1)
        self._first = True

    def __call__(self, obs):
        out = self.inner(obs)
        if self._first:
            s = out["observation.state"].detach().cpu().numpy().round(3)
            self.start_hashes += [hashlib.sha1(x.tobytes()).hexdigest()[:12] for x in s]
            self._first = False
        if self.delay == 0:
            return out
        self._hist.append(out)
        return self._hist[0]


def batch_for(episodes, cpus=None):
    """Largest simulator count within this worker's core budget that divides the episodes, so no rollout is wasted."""
    limit = max(1, cpus - 2) if cpus else int(os.environ.get("PTP_SLOTS", 0)) or max(1, parallel.available_cpus() - 2)
    for b in (25, 20, 10, 5, 2, 1):
        if b <= limit and episodes % b == 0:
            return b
    return 1


def run_cell(cfg, policy, pre, post, proto, suite, task, method, condition, episodes, seed, on_chunk=None):
    """One method, one task, one condition. Returns per-episode rows plus plan metrics."""
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    policy.config.n_action_steps = method["L"]
    env_cfg = make_env_cfg(proto, suite, [task])
    batch = batch_for(episodes)
    envs = make_env(env_cfg, n_envs=batch, use_async_envs=batch > 1)
    env = envs[suite][task]
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    hook = EnvPre(env_pre, delay=4 if condition == "delay4" else 0)
    noise_seed = int.from_bytes(hashlib.sha1(f"{seed}|{suite}|{task}|{condition}".encode()).digest()[:4], "little")
    carry = NoiseCarry(policy, method["rho"], method["mode"], seed=noise_seed, on_reset=hook.reset, on_chunk=on_chunk)
    t = time.time()
    try:
        with torch.no_grad():
            info = eval_policy(env=env, policy=policy, env_preprocessor=hook, env_postprocessor=env_post, preprocessor=pre,
                               postprocessor=post, n_episodes=episodes, start_seed=seed)
    finally:
        carry.uninstall()
        close_envs(envs)
    if carry.alignment_discards:
        raise RuntimeError(f"{carry.alignment_discards} anchors were discarded: queue accounting is wrong")
    rows = [dict(suite=suite, task=task, condition=condition, seed=seed, episode=e["episode_ix"], success=bool(e["success"]),
                 start_hash=hook.start_hashes[e["episode_ix"]]) for e in info["per_episode"]]
    return rows, dict(carry.metrics.summary(), seconds=round(time.time() - t, 1), proposals=carry.proposals, batch=batch,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)


def append_cell(path, cell):
    """Safe from several processes: a lock keeps each line whole, and fsync puts it on disk before returning."""
    with open(path, "a") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(cell) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Store:
    """Append-only results. Workers save each cell themselves, so losing any process loses at most its cell in progress."""

    def __init__(self, path):
        self.path = Path(path)
        self.cells = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a machine crash. That cell simply runs again.
                self.cells[c["key"]] = c

    def add(self, key, rows, meta):
        c = dict(key=key, rows=rows, meta=meta)
        append_cell(self.path, c)
        self.cells[key] = c
        return c

    def run(self, key, fn):
        if key in self.cells:
            return self.cells[key]
        return self.add(key, *fn())

    def rows(self, phase, names):
        out = []
        for c in self.cells.values():
            parts = c["key"].split("|")
            if parts[0] == phase and parts[1] in names:
                out += [dict(r, method=parts[1]) for r in c["rows"]]
        return out

    def meta(self, phase, name):
        return [c["meta"] for c in self.cells.values() if c["key"].split("|")[:2] == [phase, name]]


def environment_record(cfg):
    import lerobot

    return dict(lerobot=lerobot.__version__, torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cpus=os.cpu_count(),
                chunk_size=cfg.chunk_size, max_action_dim=cfg.max_action_dim, num_steps=cfg.num_steps, mujoco_gl=os.environ.get("MUJOCO_GL"))


def cmd_verify(a, proto, built):
    """Real-model checks. No result is meaningful until this passes on the rented machine."""
    cfg, policy, pre, post = built
    rec = environment_record(cfg)
    assert (cfg.chunk_size, cfg.max_action_dim, cfg.num_steps) == (50, 32, 10), rec
    log("checkpoint", rec)
    H, D = cfg.chunk_size, cfg.max_action_dim
    captured = []

    def grab(batch, chunk):
        if not captured:
            captured.append({k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)})

    rows, meta = run_cell(cfg, policy, pre, post, proto, "libero_10", 0, dict(L=10, mode="cyc", rho=0.85), "clean", 2, 1000, on_chunk=grab)
    log("short rollout", rows, meta)
    assert meta["proposals"] > 1, "expected several replans"
    assert meta["overlap_rms"] is not None and np.isfinite(meta["overlap_rms"])

    batch = {k: v[:1] for k, v in captured[0].items()}
    policy.config.n_action_steps = 10
    policy.reset()
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(1, H, D, generator=gen)
    a1, a2 = policy.predict_action_chunk(dict(batch), noise=z), policy.predict_action_chunk(dict(batch), noise=z)
    a3 = policy.predict_action_chunk(dict(batch), noise=torch.randn(1, H, D, generator=gen))
    det = float((a1 - a2).abs().max())
    assert det < 1e-3, f"same noise gave different actions, max diff {det}"
    change = float((a1 - a3).abs().mean())
    assert a1.shape == (1, H, 7), a1.shape
    assert change > 0, "different noise gave identical actions, so noise is not reaching the model"
    z0 = carry_source(z, 0.0, 10, torch.Generator().manual_seed(7))
    eps = torch.randn(1, H, D, generator=torch.Generator().manual_seed(7))
    assert torch.equal(z0, eps), "rho=0 must equal the same fresh draw"
    zp = z.clone()
    zp[:, :, 7:] = torch.randn(1, H, D - 7, generator=gen)
    pad_effect = float((policy.predict_action_chunk(dict(batch), noise=zp) - a1).abs().mean())
    out = dict(rec, determinism_max_diff=det, noise_change_mean=change, padded_noise_effect_mean=pad_effect, rollout=meta, verified=True)
    save_json(Path(a.out, "verify.json"), out)
    log("VERIFY PASSED", out)


def real_runner(proto, device):
    """Builds one model inside a worker. Runs in the worker process, after its GPU has been pinned."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    built = build(proto, device)

    def runner(c):
        rows, meta = run_cell(*built, proto, c["suite"], c["task"], c["method"], c["condition"], c["episodes"], c["seed"])
        append_cell(c["store"], dict(key=c["key"], rows=rows, meta=meta))
        return rows, dict(meta, saved_by_worker=True)

    return runner


def make_cell(key, suite, task, method, condition, episodes, seed):
    return dict(key=key, suite=suite, task=task, method=method, condition=condition, episodes=episodes, seed=seed)


def run_cells(cells, store, pool, label):
    """Runs the cells not yet stored and logs throughput and time left."""
    todo = [dict(c, store=str(store.path)) for c in cells if c["key"] not in store.cells]
    log(label, "cells", len(cells), "to run", len(todo))
    if not todo:
        return
    t0, finished = time.time(), [0]

    def on_done(key, result):
        rows, meta = result
        if meta.pop("saved_by_worker", False):
            store.cells[key] = dict(key=key, rows=rows, meta=meta)
        else:
            store.add(key, rows, meta)
        finished[0] += 1
        left = (time.time() - t0) / finished[0] * (len(todo) - finished[0]) / 3600
        log(key, "success", round(float(np.mean([r["success"] for r in rows])), 3), "| vram GB", meta.get("peak_vram_gb"), "| hours left in", label, round(left, 2))

    pool(todo, on_done)


def cmd_stage0(a, proto, built, pool):
    cfg, policy, pre, post = built
    p, store = proto["stage0"], Store(Path(a.out, "stage0_cells.jsonl"))
    H, D, L, suite = cfg.chunk_size, cfg.max_action_dim, p["L"], p["suite"]
    methods = dict(native_L50=dict(L=50, mode="native", rho=0.0), fresh_L10=dict(L=10, mode="shift", rho=0.0),
                   shift_L10_r85=dict(L=10, mode="shift", rho=0.85))
    states = []
    stride = {"n": 0, "prev": None}

    def capture(batch, chunk):
        # Sub-envs that already finished replay their last observation, so they must not count as visited states.
        s = batch["observation.state"].detach().cpu()
        moved = torch.ones(len(s), dtype=torch.bool) if stride["prev"] is None or stride["prev"].shape != s.shape else (s != stride["prev"]).any(dim=1)
        stride["prev"], stride["n"] = s, stride["n"] + 1
        if stride["n"] % 3 == 0:
            for i in torch.nonzero(moved).flatten().tolist():
                states.append({k: v[i:i + 1].detach().cpu().clone() for k, v in batch.items() if isinstance(v, torch.Tensor)})

    run_cells([make_cell(f"stage0|{n}|clean|{t}", suite, t, methods[n], "clean", p["episodes_per_task"], 1000) for t in p["tasks"] for n in ("native_L50", "shift_L10_r85")], store, pool, "stage0")
    # This model captures visited states, so these rollouts stay in the parent process.
    for task in p["tasks"]:
        store.run(f"stage0|fresh_L10|clean|{task}", lambda: run_cell(cfg, policy, pre, post, proto, suite, task, methods["fresh_L10"], "clean", p["episodes_per_task"], 1000, on_chunk=capture))
    if not states:
        # A resumed run skips the capturing rollouts, so collect states again without storing the cell.
        for task in p["tasks"]:
            run_cell(cfg, policy, pre, post, proto, suite, task, methods["fresh_L10"], "clean", p["episodes_per_task"], 1000, on_chunk=capture)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(states), min(p["states"], len(states)), replace=False)
    gen = torch.Generator().manual_seed(0)
    M, S = p["samples_per_state"], []
    resp = np.zeros((H, H))
    keep = {m: {str(r): [] for r in p["rhos"]} for m in ("shift", "same", "cyc")}
    policy.config.n_action_steps = L
    policy.reset()
    for n, i in enumerate(idx):
        st = states[i]

        def chunk_fn(z):
            b = {k: v.to(a.device).repeat(z.shape[0], *([1] * (v.ndim - 1))) for k, v in st.items()}
            return torch.cat([policy.predict_action_chunk({k: v[q:q + a.micro] for k, v in b.items()}, noise=z[q:q + a.micro]) for q in range(0, z.shape[0], a.micro)])

        z = torch.randn(M, H, D, generator=gen)
        chunks = chunk_fn(z)
        sp, sc = metrics.spread(chunks, L), metrics.mode_screen(chunks)
        S.append(dict(prefix_spread=sp["prefix"], chunk_spread=sp["chunk"], ashman=sc["ashman"], minor=sc["minor"], multimodal=sc["multimodal"]))
        if sc["multimodal"]:
            k = metrics.keep_same_observation(chunk_fn, z, sc, p["rhos"], L, gen, modes=("shift", "same", "cyc"))
            for m in k:
                for r, v in k[m].items():
                    keep[m][r].append(v)
        if n < a.response_states:
            resp += metrics.token_response(chunk_fn, H, D, gen, micro_batch=a.micro)
        if n % 10 == 0:
            log("stage0 screened", n, "of", len(idx))
    loc = metrics.locality(resp, p["locality_band"])
    succ = {n: [r["success"] for r in store.rows("stage0", {n})] for n in methods}
    rms = lambda name: float(np.sqrt(np.mean([x["overlap_rms"] ** 2 for x in store.meta("stage0", name)])))
    keep_mean = {m: {r: (float(np.mean(v)) if v else None) for r, v in d.items()} for m, d in keep.items()}
    has_keep = keep_mean["shift"]["0.95"] is not None
    prefix = np.array([s["prefix_spread"] for s in S])
    g = dict(competence=float(np.mean(succ["native_L50"])), prefix_spread_p50=float(np.median(prefix)), locality=loc["within_band"],
             keep_gain=(keep_mean["shift"]["0.95"] - keep_mean["shift"]["0.0"]) if has_keep else float("-inf"),
             jitter_gain=1 - rms("shift_L10_r85") / rms("fresh_L10"))
    decision, gates = metrics.stage0_decision(g, p["criteria"])
    out = dict(decision=decision, gates=gates, values={k: (None if v == float("-inf") else v) for k, v in g.items()},
               success={k: float(np.mean(v)) for k, v in succ.items()}, episodes={k: len(v) for k, v in succ.items()},
               spread=dict(prefix_p10=float(np.percentile(prefix, 10)), prefix_p50=float(np.median(prefix)), prefix_p90=float(np.percentile(prefix, 90))),
               multimodal_share=float(np.mean([s["multimodal"] for s in S])), states=len(S), keep_same_observation=keep_mean, locality=loc,
               token_response=[[float(x) for x in row] for row in (resp / max(resp.sum(), 1e-30))], overlap_rms={k: rms(k) for k in ("fresh_L10", "shift_L10_r85")})
    save_json(Path(a.out, "stage0_results.json"), out)
    log("STAGE 0", decision, gates, out["values"])
    return decision


def run_phase(phase, spec_tasks, names, spec, conditions, episodes, seed, store, pool):
    cells = [make_cell(f"{phase}|{n}|{c}|{s}|{t}", s, t, spec[n], c, episodes, seed)
             for c in conditions for s, ts in spec_tasks.items() for t in ts for n in names]
    run_cells(cells, store, pool, phase)


def mean_success(rows):
    """Average across tasks and conditions, so each cell counts equally."""
    cells = {}
    for r in rows:
        cells.setdefault((r["condition"], r["suite"], r["task"]), []).append(r["success"])
    return float(np.mean([np.mean(v) for v in cells.values()]))


def cmd_benchmark(a, proto, pool, env_record):
    b = proto["benchmark"]
    spec = {**b["standard"], **b["carry"]}
    store = Store(Path(a.out, "benchmark_cells.jsonl"))
    if set(sum(([(s, t) for t in ts] for s, ts in b["dev"]["tasks"].items()), [])) & set(sum(([(s, t) for t in ts] for s, ts in b["test"]["tasks"].items()), [])):
        raise ValueError("dev and test tasks must be disjoint")

    run_phase("dev", b["dev"]["tasks"], list(spec), spec, b["conditions"], b["dev"]["episodes_per_task"], b["dev"]["seed"], store, pool)
    dev_scores = {n: mean_success(store.rows("dev", {n})) for n in spec}
    ptp, standard = metrics.choose(dev_scores, list(b["carry"]), spec), metrics.choose(dev_scores, list(b["standard"]), spec)
    fresh_same_l = next(n for n, m in b["standard"].items() if m["L"] == spec[ptp]["L"])
    log("chosen on dev tasks: PTP", ptp, "standard", standard, dev_scores)
    save_json(Path(a.out, "selection.json"), dict(ptp=ptp, standard=standard, dev_success=dev_scores))

    names = list(dict.fromkeys([ptp, standard, "native_L50", fresh_same_l]))
    run_phase("test", b["test"]["tasks"], names, spec, b["conditions"], b["test"]["episodes_per_task"], b["test"]["seed"], store, pool)
    rows = store.rows("test", set(names))
    primary = metrics.paired_difference(rows, ptp, standard) if ptp != standard else None
    subset = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    base = {(r["condition"], r["suite"], r["task"], r["episode"]): r["start_hash"] for r in rows if r["method"] == standard}
    match = float(np.mean([base[(r["condition"], r["suite"], r["task"], r["episode"])] == r["start_hash"] for r in rows if r["method"] == ptp]))
    plan = {}
    for n in names:
        ms = store.meta("test", n)
        plan[n] = {k: (float(np.mean([m[k] for m in ms if m[k] is not None])) if any(m[k] is not None for m in ms) else None) for k in ("overlap_rms", "seam_rms", "executed_du_rms")}
    out = dict(
        verdict=metrics.benchmark_decision(primary, b["min_gain"]), min_gain=b["min_gain"], ptp=ptp, standard=standard,
        ptp_settings=spec[ptp], standard_settings=spec[standard], test_success={n: mean_success(subset(method=n)) for n in names},
        primary_ptp_minus_standard=primary, ptp_minus_native_L50=metrics.paired_difference(rows, ptp, "native_L50") if ptp != "native_L50" else None,
        ptp_minus_fresh_same_L=metrics.paired_difference(rows, ptp, fresh_same_l),
        by_condition={c: metrics.paired_difference(subset(condition=c), ptp, standard) for c in b["conditions"]},
        by_suite={s: metrics.paired_difference(subset(suite=s), ptp, standard) for s in b["test"]["tasks"]},
        start_state_match=match, pairing_valid=match >= 0.99, plan_consistency=plan, dev_success=dev_scores, environment=env_record, not_run=proto["not_run"])
    save_json(Path(a.out, "benchmark_results.json"), out)
    log("BENCHMARK VERDICT:", out["verdict"], "| PTP", ptp, "| standard", standard)
    log("test success", out["test_success"])
    log("PTP minus standard", primary, "| start states matched", match)
    return out


def cmd_quick(a, proto, pool, env_record):
    q = proto["quick"]
    store = Store(Path(a.out, "quick_cells.jsonl"))
    names = list(q["methods"])
    cells = [make_cell(f"quick|{n}|{c}|{s}|{t}", s, t, q["methods"][n], c, q["episodes_per_task"], q["seed"])
             for c in q["conditions"] for s, ts in q["tasks"].items() for t in ts for n in names]
    run_cells(cells, store, pool, "quick")
    rows = store.rows("quick", set(names))
    diff = lambda x, y, r=rows: metrics.paired_difference(r, x, y)
    primary = diff("same_L10_r90", "fresh_L10")
    out = dict(
        verdict=metrics.benchmark_decision(primary, q["min_gain"]), primary_same_minus_fresh=primary,
        shift_minus_fresh=diff("shift_L10_r90", "fresh_L10"), same_minus_native=diff("same_L10_r90", "native_L50"), fresh_minus_native=diff("fresh_L10", "native_L50"),
        success={n: mean_success([r for r in rows if r["method"] == n]) for n in names},
        primary_by_condition={c: diff("same_L10_r90", "fresh_L10", [r for r in rows if r["condition"] == c]) for c in q["conditions"]},
        primary_by_suite={s: diff("same_L10_r90", "fresh_L10", [r for r in rows if r["suite"] == s]) for s in q["tasks"]},
        design=q["design"], environment=env_record, not_run=proto["not_run"])
    save_json(Path(a.out, "quick_results.json"), out)
    log("QUICK VERDICT:", out["verdict"])
    log("success", out["success"])
    log("same minus fresh", primary)
    log("shift minus fresh", out["shift_minus_fresh"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "stage0", "benchmark", "all", "quick"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micro", type=int, default=32, help="GPU batch for Stage 0 noise sampling")
    ap.add_argument("--response-states", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="0 chooses from the GPUs and CPU cores found")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    proto = json.loads((HERE / "protocol.json").read_text())
    save_json(Path(a.out, "protocol_used.json"), proto)
    gpus, cpus = parallel.gpu_ids(), parallel.available_cpus()
    backend_file = HERE / ".render_backend"
    preferred = backend_file.read_text().strip() if backend_file.exists() else "egl"
    backend, good = parallel.resolve_rendering(gpus, preferred)
    mem_gb, vram_gb = parallel.available_memory_gb(), parallel.smallest_gpu_memory_gb()
    assigned, slots = parallel.plan_workers(len(gpus), good, cpus, a.workers, mem_gb, vram_gb, backend)
    log(f"render backend {backend}, GPUs {len(gpus)}, usable for this run {good}, usable cores {cpus}, free RAM GB {mem_gb and round(mem_gb)}, GPU memory GB {vram_gb and round(vram_gb)}, workers on GPUs {assigned}, simulators per worker {slots}")
    os.environ["PTP_SLOTS"] = str(slots)
    # The parent's own model pins to a GPU that can render, so its simulators work too.
    parallel.apply_pin(good[0])

    # Finished steps are reused, so a restart goes straight back to the work that is left.
    verify_file, stage0_file = Path(a.out, "verify.json"), Path(a.out, "stage0_results.json")
    want_verify = a.command == "verify" or (a.command == "all" and not verify_file.exists())
    want_stage0 = a.command == "stage0" or (a.command == "all" and not stage0_file.exists())
    built = build(proto, a.device) if want_verify or want_stage0 else None
    if built is not None:
        base_record = environment_record(built[0])
    elif verify_file.exists():
        base_record = {k: v for k, v in json.loads(verify_file.read_text()).items() if k != "rollout"}
    else:
        base_record = {}
    env_record = dict(base_record, render_backend=backend, workers=assigned, simulators_per_worker=slots, usable_cores=cpus, free_ram_gb=mem_gb)
    if want_verify:
        cmd_verify(a, proto, built)
    if a.command == "verify":
        log("FINISHED")
        return

    workers = parallel.Pool(assigned, slots, real_runner, (proto, a.device), log=log)
    try:
        decision = None
        if want_stage0:
            decision = cmd_stage0(a, proto, built, workers.run)
        elif a.command == "all" and stage0_file.exists():
            decision = json.loads(stage0_file.read_text())["decision"]
            log("reusing the finished Stage 0:", decision)
        if a.command == "all" and decision == "FIX-BASE":
            log("STOPPED: the checkpoint is not competent in this setup, so a comparison would mean nothing. See stage0_results.json.")
            return
        if decision == "NO-GO":
            log("Stage 0 says noise carry has little to act on. Continuing with the benchmark, which will measure it directly.")
        del built
        torch.cuda.empty_cache()
        if a.command in ("benchmark", "all"):
            cmd_benchmark(a, proto, workers.run, env_record)
        if a.command == "quick":
            cmd_quick(a, proto, workers.run, env_record)
    finally:
        workers.close()
    log("FINISHED")


if __name__ == "__main__":
    main()
