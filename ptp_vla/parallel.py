"""Runs benchmark cells across GPUs. One model per worker process, with dynamic scheduling and crash recovery.

Cells are deterministic given their seeds, so which worker runs a cell does not change what it measures, and a
crashed or stuck cell can safely be run again.
"""
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
import traceback


def available_cpus():
    """Cores this process may really use, respecting affinity and any container CPU quota."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def available_memory_gb():
    """RAM free for new processes, respecting a container limit when there is one."""
    free = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                free = int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        used = int(open("/sys/fs/cgroup/memory.current").read())
        if limit != "max":
            room = (int(limit) - used) / 2**30
            free = room if free is None else min(free, room)
    except (OSError, ValueError):
        pass
    return free


def gpu_ids():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=True).stdout
        return list(range(sum(line.startswith("GPU ") for line in out.splitlines())))
    except (OSError, subprocess.SubprocessError):
        return []


def smallest_gpu_memory_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True).stdout
        return min(int(x) for x in out.split()) / 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def render_backend():
    return os.environ.get("PTP_RENDER", "egl")


def pin_env(gpu):
    """Binds a worker's model to one physical GPU. With EGL, its simulators render on that GPU too."""
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": render_backend(), "PYOPENGL_PLATFORM": render_backend()}
    if render_backend() == "egl":
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    return env


def apply_pin(gpu, environ=None):
    environ = os.environ if environ is None else environ
    environ.pop("MUJOCO_EGL_DEVICE_ID", None)  # robosuite asserts this matches CUDA_VISIBLE_DEVICES when present
    environ.update(pin_env(gpu))
    return environ


def probe_rendering(gpus, timeout=300):
    """Renders a real LIBERO frame pinned to each GPU. Containers can expose fewer EGL devices than GPUs."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_test.py")
    procs = {g: subprocess.Popen([sys.executable, script], env=apply_pin(g, dict(os.environ)), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for g in gpus}
    good = []
    for g, p in procs.items():
        try:
            _, err = p.communicate(timeout=timeout)
            if p.returncode == 0:
                good.append(g)
            else:
                print(f"GPU {g} failed the render probe: {err[-400:]}", flush=True)
        except subprocess.TimeoutExpired:
            p.kill()
            print(f"GPU {g} timed out on the render probe", flush=True)
    return good


def resolve_rendering(gpus, preferred="egl", probe=probe_rendering):
    """Returns (backend, GPUs usable). Software rendering needs no GPU, so every GPU stays usable for the model."""
    os.environ["PTP_RENDER"] = preferred
    good = probe(gpus)
    if good:
        return preferred, good if preferred == "egl" else gpus
    if preferred == "egl":
        os.environ["PTP_RENDER"] = "osmesa"
        if probe(gpus[:1]):
            return "osmesa", gpus
    raise RuntimeError("no rendering backend works on this machine")


GB_PER_SIMULATOR_RAM = 3.0  # cautious. 160 simulators are the likely reason the first full run was killed
GB_PER_SIMULATOR_VRAM = 1.3  # from about 28 GB seen on a card running simulators in the first full run


def plan_workers(gpus_total, gpus_good, cpus, requested=0, mem_gb=None, vram_gb=None, backend="egl"):
    """Returns (physical GPU per worker, simulator slots per worker), limited by cores, RAM and GPU memory.

    One worker per GPU: the simulators are the bottleneck, so a second model on a card adds memory use, not speed.
    """
    if not gpus_good:
        raise RuntimeError("no GPU can render LIBERO through EGL")
    n = max(1, min(requested or gpus_total, cpus))
    assigned = [gpus_good[i % len(gpus_good)] for i in range(n)]
    total = cpus if n > 1 else max(1, cpus - 2)
    if mem_gb is not None:
        total = min(total, int(max(0.0, mem_gb - 8 * n) / GB_PER_SIMULATOR_RAM))
    if vram_gb is not None and backend == "egl":
        total = min(total, int(vram_gb * len(gpus_good) / GB_PER_SIMULATOR_VRAM))
    return assigned, max(1, total // n)


def _worker(wid, gpu, slots, factory, factory_args, tasks, results):
    if hasattr(os, "setsid"):
        os.setsid()  # gives this worker and its simulators one process group, so a stuck cell can be killed whole
    if gpu is not None:
        apply_pin(gpu)
    os.environ["PTP_SLOTS"] = str(slots)
    try:
        runner = factory(*factory_args)
        results.put(("ready", wid, None, None))
        parent = os.getppid()
        while True:
            if hasattr(os, "getppid") and os.getppid() != parent:
                return  # the parent died: stop instead of running unattended. Finished cells are already on disk.
            try:
                cell = tasks.get(timeout=10)
            except queue.Empty:
                continue
            if cell is None:
                return
            results.put(("claim", wid, cell["key"], None))
            try:
                results.put(("done", wid, cell["key"], runner(cell)))
            except Exception:
                results.put(("error", wid, cell["key"], traceback.format_exc()))
    except Exception:
        results.put(("fatal", wid, None, traceback.format_exc()))


def _kill_tree(proc):
    if proc.pid is None or not proc.is_alive():
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.join(timeout=10)


class Pool:
    def __init__(self, gpus, slots, factory, factory_args=(), max_attempts=3, cell_timeout=40 * 60, poll=2.0, log=print):
        self.ctx = mp.get_context("spawn")
        self.gpus, self.slots, self.factory, self.factory_args = gpus, slots, factory, factory_args
        self.max_attempts, self.cell_timeout, self.poll, self.log = max_attempts, cell_timeout, poll, log
        self.tasks, self.results = self.ctx.Queue(), self.ctx.Queue()
        self.procs, self.claims = {}, {}
        self.restarts = 0
        for wid in range(len(gpus)):
            self._start(wid)

    def _start(self, wid):
        p = self.ctx.Process(target=_worker, args=(wid, self.gpus[wid], self.slots, self.factory, self.factory_args, self.tasks, self.results), daemon=False)
        p.start()
        self.procs[wid] = p
        self.claims.pop(wid, None)

    def run(self, cells, on_done):
        """Runs cells to completion, calling on_done(key, result) in the parent. Raises only if a cell keeps failing."""
        by_key = {c["key"]: c for c in cells}
        attempts = {k: 0 for k in by_key}
        pending = set(by_key)
        for c in cells:
            self.tasks.put(c)

        def failed(key, why):
            attempts[key] += 1
            if attempts[key] >= self.max_attempts:
                raise RuntimeError(f"cell {key} failed {attempts[key]} times. Last error:\n{why}")
            self.log(f"retrying {key} (attempt {attempts[key] + 1}): {str(why)[-300:]}")
            self.tasks.put(by_key[key])

        while pending:
            try:
                kind, wid, key, payload = self.results.get(timeout=self.poll)
            except queue.Empty:
                self._check_workers(pending, failed)
                continue
            if kind == "claim":
                self.claims[wid] = (key, time.time())
            elif kind == "done":
                self.claims.pop(wid, None)
                if key in pending:
                    pending.discard(key)
                    on_done(key, payload)
            elif kind == "error":
                self.claims.pop(wid, None)
                if key in pending:
                    failed(key, payload)
            elif kind == "fatal":
                raise RuntimeError(f"worker {wid} could not start:\n{payload}")

    def _check_workers(self, pending, failed):
        now = time.time()
        for wid, p in list(self.procs.items()):
            claim = self.claims.get(wid)
            if claim and now - claim[1] > self.cell_timeout:
                self.log(f"worker {wid} stuck on {claim[0]} for over {self.cell_timeout} s, killing it")
                _kill_tree(p)
            if not p.is_alive():
                self.restarts += 1
                if self.restarts > 20:
                    raise RuntimeError("too many worker crashes")
                self.log(f"worker {wid} exited with code {p.exitcode}, restarting it")
                self._start(wid)
                if claim and claim[0] in pending:
                    failed(claim[0], f"worker died with exit code {p.exitcode}")

    def close(self):
        for _ in self.procs:
            self.tasks.put(None)
        deadline = time.time() + 30
        for p in self.procs.values():
            p.join(timeout=max(0.1, deadline - time.time()))
            _kill_tree(p)
        self.procs.clear()
