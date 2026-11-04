"""Benchmark plumbing: pairing, events, delay, ensembling and the declared decision rules."""
import numpy as np
import pytest
import torch

import bench
import run_all
from model import FMPolicy

SEEDS = list(range(2_000_000, 2_000_016))


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    m = FMPolicy(width=32, layers=1, heads=2, ff=64).eval()
    return m


def test_scenes_and_events_depend_only_on_seed():
    a, b = bench.EventEnv(SEEDS, "goal"), bench.EventEnv(SEEDS, "goal")
    assert np.array_equal(a.x, b.x) and np.array_equal(a.goal, b.goal) and np.array_equal(a.te, b.te)
    assert np.array_equal(a.mag, b.mag)


@pytest.mark.parametrize("cond,field", [("goal", "goal"), ("velocity", "v"), ("obstacle", "obst")])
def test_event_fires_once_at_its_tick(cond, field):
    e = bench.EventEnv(SEEDS[:1], cond)
    e.te[:] = 3
    e.x[:] = [-0.8, 0.0]
    before = None
    changes = 0
    for t in range(6):
        if t == 3:
            before = getattr(e, field).copy()
        e.step(np.zeros((1, 2)))
        if t == 3:
            changes = int(not np.allclose(before, getattr(e, field)))
    assert changes == 1


def test_clean_has_no_event():
    e = bench.EventEnv(SEEDS[:2], "clean")
    g, o = e.goal.copy(), e.obst.copy()
    for _ in range(60):
        e.step(np.zeros((2, 2)))
    assert np.array_equal(g, e.goal) and np.array_equal(o, e.obst)


def test_unsafe_obstacle_move_is_a_counted_noop():
    e = bench.EventEnv(SEEDS[:1], "obstacle")
    e.te[:] = 0
    e.obst[:] = [-0.5, 0.0]
    e.x[:] = [-0.5, 0.25]
    e.mag[:] = 0.2
    e.jit[:] = 0.0
    o = e.obst.copy()
    e._apply_events()
    assert e.event_noop.all() and np.array_equal(o, e.obst)


def test_delay_serves_observation_from_two_ticks_ago():
    e = bench.EventEnv(SEEDS[:1], "delay2")
    e.record_tick()
    first = e.current_policy_obs().copy()
    u = np.ones((1, 2))
    obs = []
    for _ in range(3):
        e.step(u)
        e.record_tick()
        obs.append(e.current_policy_obs().copy())
    assert np.array_equal(obs[0], first) and np.array_equal(obs[1], first)
    assert not np.array_equal(obs[2], first)


def test_rollout_outputs_and_rho0_is_seed_identical_for_shift_and_same(policy):
    r1 = bench.rollout(policy, SEEDS, "clean", 4, 0.0, "shift", gen_seed=5)
    r2 = bench.rollout(policy, SEEDS, "clean", 4, 0.0, "same", gen_seed=5)
    assert np.array_equal(r1["finish_t"], r2["finish_t"]) and r1["du_rms"] == r2["du_rms"]


@pytest.mark.parametrize("kw", [dict(L=2), dict(L=8, rho=0.85), dict(L=4, rho=0.95, mode="cyc"), dict(L=1, ensemble=True)])
@pytest.mark.parametrize("cond", ["goal", "delay2"])
def test_every_method_runs_every_kind_of_condition(policy, kw, cond):
    r = bench.rollout(policy, SEEDS[:6], cond, **kw)
    assert len(r["success"]) == 6 and np.isfinite(r["du_rms"])


def _cells(name, succ):
    return {f"0|clean|{name}": dict(success=list(succ))}


def test_selection_rule_prefers_fresh_when_differences_are_noise():
    rng = np.random.default_rng(0)
    val = _cells("shift_0.0", rng.integers(0, 2, 1000))
    val.update(_cells("shift_0.5", rng.integers(0, 2, 1000)))
    s = run_all.select(val, {"shift": [0.5]}, ["clean"], [0])
    assert s["shift"]["rho"] in (0.0, 0.5)
    base = np.zeros(1000, int); better = base.copy(); better[:300] = 1
    val = _cells("shift_0.0", base); val.update(_cells("shift_0.5", better)); val.update(_cells("shift_0.85", better))
    s = run_all.select(val, {"shift": [0.5, 0.85]}, ["clean"], [0])
    assert s["shift"]["rho"] == 0.5, "the lowest rho tied with the best must win"


@pytest.mark.parametrize("rho,diff,ci,seeds,want", [
    (0.0, 0.1, [0.05, 0.15], [0.1], "NULL"), (0.5, 0.05, [0.01, 0.09], [0.04, 0.05, 0.06], "WIN"),
    (0.5, 0.02, [0.01, 0.03], [0.02, 0.02, 0.02], "NO-WIN"), (0.5, 0.05, [-0.01, 0.1], [0.05, 0.05, 0.05], "NO-WIN"),
    (0.5, 0.05, [0.01, 0.09], [0.1, 0.1, -0.05], "NO-WIN")])
def test_verdict_rule(rho, diff, ci, seeds, want):
    v = run_all.verdict(dict(shift=dict(rho=rho)), dict(shift_sel=dict(diff=diff, ci95=ci, by_seed=seeds)))
    assert v.startswith(want)
