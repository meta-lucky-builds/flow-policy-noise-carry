"""Real worker processes with a fake cell runner: scheduling, crashes, hangs and failures."""
import os
import time
from pathlib import Path

import pytest

import parallel


def fake_factory(mode, marker_dir):
    """Top-level so spawned workers can import it."""
    if mode == "cannot_start":
        raise RuntimeError("model failed to load")
    marks = Path(marker_dir)

    def once(key):
        f = marks / key
        first = not f.exists()
        f.write_text("x")
        return first

    def runner(c):
        key = c["key"]
        if mode == "crash_once" and key == "b" and once(key):
            os._exit(9)
        if mode == "raise_once" and key == "b" and once(key):
            raise ValueError("transient")
        if mode == "always_raise" and key == "b":
            raise ValueError("permanent")
        if mode == "hang_once" and key == "b" and once(key):
            time.sleep(600)
        time.sleep(c.get("sleep", 0))
        return [dict(key=key, pid=os.getpid(), slots=os.environ.get("PTP_SLOTS"), cvd=os.environ.get("CUDA_VISIBLE_DEVICES"), egl=os.environ.get("MUJOCO_EGL_DEVICE_ID"))], {}

    return runner


def run_pool(tmp_path, mode, keys="abcdef", gpus=(0, 1), **kw):
    pool = parallel.Pool(list(gpus), 7, fake_factory, (mode, str(tmp_path)), poll=0.2, log=lambda *a: None, **kw)
    got = {}
    try:
        pool.run([dict(key=k, sleep=0.2) for k in keys], lambda k, r: got.__setitem__(k, r))
    finally:
        pool.close()
    return got, pool


def test_all_cells_finish_once_on_multiple_pinned_workers(tmp_path):
    got, _ = run_pool(tmp_path, "ok", "abcdefgh")
    assert sorted(got) == list("abcdefgh")
    rows = [r[0][0] for r in got.values()]
    assert len({r["pid"] for r in rows}) == 2, "both workers must share the work"
    assert {(r["cvd"], r["egl"]) for r in rows} == {("0", "0"), ("1", "1")} and {r["slots"] for r in rows} == {"7"}


def test_worker_crash_is_recovered_without_losing_a_cell(tmp_path):
    got, pool = run_pool(tmp_path, "crash_once")
    assert sorted(got) == list("abcdef") and pool.restarts == 1


def test_cell_error_is_retried(tmp_path):
    got, pool = run_pool(tmp_path, "raise_once")
    assert sorted(got) == list("abcdef") and pool.restarts == 0


def test_stuck_cell_is_killed_and_rerun(tmp_path):
    got, pool = run_pool(tmp_path, "hang_once", cell_timeout=3)
    assert sorted(got) == list("abcdef") and pool.restarts == 1


def test_permanent_failure_stops_the_run_and_keeps_finished_results(tmp_path):
    pool = parallel.Pool([0], 7, fake_factory, ("always_raise", str(tmp_path)), poll=0.2, log=lambda *a: None)
    got = {}
    try:
        with pytest.raises(RuntimeError, match="cell b failed 3 times"):
            pool.run([dict(key=k) for k in "abc"], lambda k, r: got.__setitem__(k, r))
    finally:
        pool.close()
    assert "a" in got


def test_worker_that_cannot_start_is_reported(tmp_path):
    pool = parallel.Pool([0], 7, fake_factory, ("cannot_start", str(tmp_path)), poll=0.2, log=lambda *a: None)
    try:
        with pytest.raises(RuntimeError, match="could not start"):
            pool.run([dict(key="a")], lambda k, r: None)
    finally:
        pool.close()


def test_pool_is_reusable_across_phases(tmp_path):
    pool = parallel.Pool([0], 7, fake_factory, ("ok", str(tmp_path)), poll=0.2, log=lambda *a: None)
    got = {}
    try:
        pool.run([dict(key="a")], lambda k, r: got.__setitem__(k, r))
        pool.run([dict(key="b")], lambda k, r: got.__setitem__(k, r))
    finally:
        pool.close()
    assert sorted(got) == ["a", "b"]


@pytest.mark.parametrize("total,good,cpus,mem,vram,backend,want_gpus,want_slots", [
    (4, [0, 1, 2, 3], 256, 500, 32, "egl", [0, 1, 2, 3], 24),  # the machine that was killed: GPU memory now sets the limit
    (4, [0, 1, 2, 3], 256, 150, 32, "egl", [0, 1, 2, 3], 9),  # less RAM means fewer simulators, not a crash
    (4, [0, 1, 2, 3], 96, None, None, "egl", [0, 1, 2, 3], 24),
    (1, [0], 16, None, None, "egl", [0], 14),
    (1, [0], 96, 500, 32, "egl", [0], 24),
    (4, [0], 96, 500, 32, "egl", [0, 0, 0, 0], 6),  # only one card can render, so its GPU memory is shared
    (4, [0, 1, 2, 3], 256, 1000, 32, "osmesa", [0, 1, 2, 3], 64),  # software rendering needs no GPU memory
    (2, [0, 1], 8, None, None, "egl", [0, 1], 4),
])
def test_worker_plan(total, good, cpus, mem, vram, backend, want_gpus, want_slots):
    assert parallel.plan_workers(total, good, cpus, 0, mem, vram, backend) == (want_gpus, want_slots)


def test_low_memory_still_gives_one_simulator():
    assert parallel.plan_workers(4, [0, 1, 2, 3], 256, 4, 32) == ([0, 1, 2, 3], 1)


def test_egl_pin_matches_robosuite_requirement(monkeypatch):
    monkeypatch.setenv("PTP_RENDER", "egl")
    e = parallel.apply_pin(2, {"MUJOCO_EGL_DEVICE_ID": "0"})
    assert e["CUDA_VISIBLE_DEVICES"] == "2" and e["MUJOCO_EGL_DEVICE_ID"] == "2" and e["MUJOCO_GL"] == "egl" and e["PYOPENGL_PLATFORM"] == "egl"


def test_osmesa_pin_removes_egl_device(monkeypatch):
    monkeypatch.setenv("PTP_RENDER", "osmesa")
    e = parallel.apply_pin(2, {"MUJOCO_EGL_DEVICE_ID": "0"})
    assert e["CUDA_VISIBLE_DEVICES"] == "2" and "MUJOCO_EGL_DEVICE_ID" not in e and e["MUJOCO_GL"] == "osmesa" and e["PYOPENGL_PLATFORM"] == "osmesa"


def test_rendering_prefers_egl_and_keeps_only_cards_that_pass(monkeypatch):
    monkeypatch.delenv("PTP_RENDER", raising=False)
    assert parallel.resolve_rendering([0, 1, 2, 3], "egl", probe=lambda g: [0, 2]) == ("egl", [0, 2])


def test_rendering_falls_back_to_cpu_and_keeps_all_gpus_for_the_model(monkeypatch):
    monkeypatch.delenv("PTP_RENDER", raising=False)
    probe = lambda g: g if parallel.render_backend() == "osmesa" else []
    assert parallel.resolve_rendering([0, 1, 2, 3], "egl", probe=probe) == ("osmesa", [0, 1, 2, 3])
    assert parallel.render_backend() == "osmesa"


def test_rendering_recorded_as_osmesa_is_used_directly(monkeypatch):
    monkeypatch.delenv("PTP_RENDER", raising=False)
    seen = []
    assert parallel.resolve_rendering([0, 1], "osmesa", probe=lambda g: seen.append(parallel.render_backend()) or g) == ("osmesa", [0, 1])
    assert seen == ["osmesa"]


def test_rendering_failure_is_reported(monkeypatch):
    monkeypatch.delenv("PTP_RENDER", raising=False)
    with pytest.raises(RuntimeError, match="no rendering backend"):
        parallel.resolve_rendering([0], "egl", probe=lambda g: [])


def test_plan_refuses_when_no_gpu_can_render():
    with pytest.raises(RuntimeError, match="no GPU"):
        parallel.plan_workers(4, [], 96)
