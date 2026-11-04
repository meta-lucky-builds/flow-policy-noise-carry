"""Runs SmolVLAPolicy's real select_action, _get_action_chunk and reset. Only the neural network is replaced."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

import carry as C
import metrics as M

H, D, A = 50, 32, 7


class FakeModel:
    """Same call contract as VLAFlowMatching.sample_actions. Records exactly the noise it receives."""

    def __init__(self):
        self.seen = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        if noise is None:
            noise = torch.randn(state.shape[0], H, D)
            self.seen.append(("internal", noise.clone()))
        else:
            self.seen.append(("supplied", noise.clone()))
        return noise * 0.5


def make_policy(L):
    p = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(p)
    p.config = SimpleNamespace(chunk_size=H, n_action_steps=L, max_action_dim=D, adapt_to_pi_aloha=False, rtc_config=None,
                               action_feature=SimpleNamespace(shape=(A,)))
    p.model = FakeModel()
    p.prepare_images = lambda batch: ([], [])
    p.prepare_state = lambda batch: batch[OBS_STATE]
    p.reset()
    return p


def batch(B=2, value=None):
    s = torch.rand(B, 8) if value is None else torch.full((B, 8), float(value))
    return {OBS_STATE: s, OBS_LANGUAGE_TOKENS: torch.zeros(B, 4).long(), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(B, 4).bool()}


def drive(p, steps, B=2):
    return [p.select_action(batch(B)) for _ in range(steps)]


def test_real_queue_refills_every_L_steps_and_uses_supplied_noise():
    p = make_policy(10)
    c = C.NoiseCarry(p, 0.85, "shift", seed=1)
    out = drive(p, 25)
    assert out[0].shape == (2, A)
    assert [k for k, _ in p.model.seen] == ["supplied"] * 3
    assert c.proposals == 3 and c.alignment_discards == 0


def test_noise_has_checkpoint_shape_including_padded_dims():
    p = make_policy(10)
    C.NoiseCarry(p, 0.5, "shift", seed=1)
    drive(p, 1, B=3)
    assert p.model.seen[0][1].shape == (3, H, D)


def test_shift_law_through_real_policy():
    p = make_policy(10)
    C.NoiseCarry(p, 0.95, "shift", seed=2)
    drive(p, 11, B=20000)
    z0, z1 = p.model.seen[0][1], p.model.seen[1][1]
    corr = float((z1[:, :40] * z0[:, 10:]).mean())
    assert corr == pytest.approx(0.95, abs=0.01)
    assert float((z1[:, :40] * z0[:, :40]).mean()) == pytest.approx(0, abs=0.01)
    assert float((z1[:, 40:] * z0[:, :10]).mean()) == pytest.approx(0, abs=0.01)
    assert float(z1.var()) == pytest.approx(1, abs=0.01)


def test_same_law_through_real_policy():
    p = make_policy(10)
    C.NoiseCarry(p, 0.85, "same", seed=3)
    drive(p, 11, B=20000)
    z0, z1 = p.model.seen[0][1], p.model.seen[1][1]
    assert float((z1 * z0).mean()) == pytest.approx(0.85, abs=0.01)


def test_rho0_matches_independent_draws():
    p = make_policy(10)
    C.NoiseCarry(p, 0.0, "shift", seed=4)
    drive(p, 11, B=20000)
    z0, z1 = p.model.seen[0][1], p.model.seen[1][1]
    assert float((z1[:, :40] * z0[:, 10:]).mean()) == pytest.approx(0, abs=0.01)


def test_native_mode_leaves_sampling_to_the_checkpoint():
    p = make_policy(50)
    C.NoiseCarry(p, 0.0, "native", seed=5)
    drive(p, 51)
    assert [k for k, _ in p.model.seen] == ["internal"] * 2


def test_shift_carry_refuses_native_L50():
    with pytest.raises(ValueError, match="needs overlap"):
        C.NoiseCarry(make_policy(50), 0.85, "shift")


def test_rollout_reset_clears_memory_and_queue():
    p = make_policy(10)
    resets = []
    c = C.NoiseCarry(p, 0.95, "shift", seed=6, on_reset=lambda: resets.append(1))
    drive(p, 15, B=20000)
    p.reset()
    assert resets == [1] and c._z is None and len(p._queues[ACTION]) == 0
    drive(p, 1, B=20000)
    z_prev, z_new = p.model.seen[-2][1], p.model.seen[-1][1]
    assert float((z_new[:, :40] * z_prev[:, 10:]).mean()) == pytest.approx(0, abs=0.01)


def test_queue_tampering_discards_anchor():
    p = make_policy(10)
    c = C.NoiseCarry(p, 0.95, "shift", seed=7)
    drive(p, 10)
    c._pops = 7
    drive(p, 1)
    assert c.alignment_discards == 1


def test_uninstall_restores_the_original_methods():
    p = make_policy(10)
    c = C.NoiseCarry(p, 0.5, "shift")
    assert "select_action" in p.__dict__
    c.uninstall()
    assert "select_action" not in p.__dict__ and p.select_action.__func__ is SmolVLAPolicy.select_action


def test_rejects_rho_above_cap_and_outside_noise():
    with pytest.raises(ValueError):
        C.carry_source(torch.zeros(1, H, D), 0.96, 10, torch.Generator())
    p = make_policy(10)
    C.NoiseCarry(p, 0.5, "shift")
    with pytest.raises(ValueError, match="owns the noise"):
        p.select_action(batch(), noise=torch.zeros(2, H, D))


def test_metrics_ignore_frozen_envs_and_measure_overlap():
    m = C.ChunkMetrics()
    a = torch.zeros(2, 20, 7)
    s = torch.zeros(2, 8)
    m.update(a, s, 5)
    b = torch.ones(2, 20, 7)
    s2 = s.clone()
    s2[0] += 1  # env 0 moved, env 1 frozen
    m.update(b, s2, 5)
    assert m.seam_n == 1 and m.overlap_n == 15 * 7
    assert m.summary()["overlap_rms"] == pytest.approx(1.0)


class TwoMode:
    """Exact flow for two isotropic Gaussian modes with noise_at_one convention irrelevant to the map."""

    def __init__(self, H=50, A=7, sep=1.0, s=0.1):
        self.m, self.s2, self.H, self.A = torch.ones(H, A) * sep, s**2, H, A

    def __call__(self, z):
        x = z[:, :, : self.A].double()
        m = self.m.double()
        for i in range(40):
            t = i / 40
            c = (1 - t) ** 2 + t**2 * self.s2
            ra, rb = x - t * m, x + t * m
            wa = torch.sigmoid((-(ra**2).sum((1, 2)) + (rb**2).sum((1, 2))) / (2 * c))[:, None, None]
            g = (t * self.s2 - (1 - t)) / c
            x = x + (wa * (m + g * ra) + (1 - wa) * (-m + g * rb)) / 40
        return x.float()


def test_multimodal_screen_rejects_gaussian_and_finds_two_modes():
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(256, H, D, generator=gen)
    assert not M.mode_screen(z[:, :, :A] * 0.1)["multimodal"]
    assert M.mode_screen(TwoMode()(z))["multimodal"]


def test_keep_rate_matches_analytic_answer_at_H50_L10():
    gen = torch.Generator().manual_seed(1)
    f = TwoMode()
    z = torch.randn(1500, H, D, generator=gen)
    sc = M.mode_screen(f(z))
    k = M.keep_same_observation(f, z, sc, [0.0, 0.95], 10, gen)
    assert k["shift"]["0.0"] == pytest.approx(0.5, abs=0.04)
    assert k["shift"]["0.95"] == pytest.approx(0.5 + np.arcsin(0.8 * 0.95) / np.pi, abs=0.04)
    assert k["same"]["0.95"] == pytest.approx(0.5 + np.arcsin(0.95) / np.pi, abs=0.04)


def test_token_response_recovers_known_local_and_global_maps():
    gen = torch.Generator().manual_seed(2)
    local = lambda z: 0.5 * z[:, :, :A]
    r = M.token_response(local, H, D, gen, directions=2)
    assert M.locality(r, 5)["within_band"] > 0.999
    W = torch.randn(H, H, generator=torch.Generator().manual_seed(3))
    glob = lambda z: W @ z[:, :, :A]
    r2 = M.token_response(glob, H, D, gen, directions=4)
    lg = M.locality(r2, 5)
    assert lg["within_band"] == pytest.approx(lg["null_uniform"], abs=0.1)


def _rows(a, b, tasks=4, eps=50):
    rows = []
    for t in range(tasks):
        for e in range(eps):
            for name, p in (("a", a), ("b", b)):
                rows.append(dict(method=name, condition="clean", suite="s", task=t, seed=0, episode=e, success=bool((e * 7 % eps) / eps < p)))
    return rows


def test_paired_difference_exact_and_refuses_unpaired():
    d = M.paired_difference(_rows(0.6, 0.4), "a", "b")
    assert d["diff"] == pytest.approx(0.2) and d["n_episodes"] == 200 and d["ci95"][0] <= 0.2 <= d["ci95"][1]
    rows = _rows(0.6, 0.4)[:-1]
    with pytest.raises(ValueError, match="not fully paired"):
        M.paired_difference(rows, "a", "b")


@pytest.mark.parametrize("v,want", [
    (dict(competence=0.3, prefix_spread_p50=1, locality=1, keep_gain=1, jitter_gain=1), "FIX-BASE"),
    (dict(competence=0.8, prefix_spread_p50=0.01, locality=1, keep_gain=1, jitter_gain=1), "NO-GO"),
    (dict(competence=0.8, prefix_spread_p50=0.2, locality=0.2, keep_gain=1, jitter_gain=1), "NO-GO"),
    (dict(competence=0.8, prefix_spread_p50=0.2, locality=0.8, keep_gain=0, jitter_gain=0), "NO-GO"),
    (dict(competence=0.8, prefix_spread_p50=0.2, locality=0.8, keep_gain=float("-inf"), jitter_gain=0.3), "GO"),
    (dict(competence=0.8, prefix_spread_p50=0.2, locality=0.8, keep_gain=0.2, jitter_gain=0), "GO")])
def test_stage0_rule(v, want):
    crit = dict(G0_competence_min=0.5, G1_spread_min=0.05, G2_locality_min=0.6, G3_keep_gain_min=0.1, G4_jitter_gain_min=0.2)
    assert M.stage0_decision(v, crit)[0] == want


@pytest.mark.parametrize("diff,ci,fixed,want", [
    (0.05, [0.01, 0.1], [0.02, 0.08], "WIN"), (0.05, [-0.02, 0.1], [0.02, 0.08], "WIN-ON-THESE-TASKS-ONLY"),
    (0.05, [-0.02, 0.1], [-0.01, 0.1], "NO-IMPROVEMENT-SHOWN"), (0.02, [0.01, 0.03], [0.01, 0.03], "NO-IMPROVEMENT-SHOWN"),
    (-0.05, [-0.1, 0.01], [-0.08, -0.02], "WORSE-THAN-STANDARD")])
def test_benchmark_rule(diff, ci, fixed, want):
    assert M.benchmark_decision(dict(diff=diff, ci95=ci, ci95_these_tasks=fixed), 0.03) == want


def test_cyc_law_keeps_alignment_recycles_tail_and_stays_standard_normal():
    p = make_policy(10)
    C.NoiseCarry(p, 0.9, "cyc", seed=8)
    drive(p, 11, B=20000)
    z0, z1 = p.model.seen[0][1], p.model.seen[1][1]
    assert float((z1[:, :40] * z0[:, 10:]).mean()) == pytest.approx(0.9, abs=0.01)
    assert float((z1[:, 40:] * z0[:, :10]).mean()) == pytest.approx(0.9, abs=0.01)
    assert float(z1.var()) == pytest.approx(1, abs=0.01)
    f = z1[:, ::10, 0]
    assert torch.allclose(torch.cov(f.T), torch.eye(5), atol=0.04)


def test_cyc_memory_outlasts_shift():
    gen = torch.Generator().manual_seed(9)
    z0 = torch.randn(40000, H, 1, generator=gen)
    a, b = z0, z0
    for _ in range(5):
        a = C.carry_source(a, 0.9, 10, gen, "shift")
        b = C.carry_source(b, 0.9, 10, gen, "cyc")
    assert abs(float((a * z0).mean())) < 0.01
    assert abs(float((a[:, :40] * z0[:, 10:]).mean())) < 0.01
    assert float((b * z0).mean()) == pytest.approx(0.9**5, abs=0.015), "after 5 cyclic shifts of 10, tokens return home"


def test_batch_size_divides_episodes_and_respects_cores():
    import run

    assert run.batch_for(20, 32) == 20 and run.batch_for(50, 32) == 25 and run.batch_for(50, 16) == 10
    assert run.batch_for(20, 8) == 5 and run.batch_for(10, 4) == 2 and run.batch_for(5, 1) == 1
    assert run.batch_for(50, 3) == 1


class SimWorld:
    """Replaces build and run_cell with a simulated success table, so selection and verdicts have known answers."""

    def __init__(self, truth):
        self.truth, self.calls = truth, []

    def run_cell(self, cfg, policy, pre, post, proto, suite, task, method, condition, episodes, seed, on_chunk=None):
        name = next(n for n, m in {**proto["benchmark"]["standard"], **proto["benchmark"]["carry"]}.items() if m == method)
        self.calls.append((name, suite, task, condition, seed, episodes))
        rng = np.random.default_rng(abs(hash((suite, task, condition, seed))) % 2**32)
        u = rng.random(episodes)  # common random numbers, so paired differences are exact
        rows = [dict(suite=suite, task=task, condition=condition, seed=seed, episode=i, success=bool(u[i] < self.truth(name)), start_hash=str(i)) for i in range(episodes)]
        return rows, dict(overlap_rms=None if method["L"] == 50 else 0.1, seam_rms=0.1, executed_du_rms=0.1, seconds=1.0, peak_vram_gb=None)


def run_sim(monkeypatch, tmp_path, truth):
    import json as js

    import run

    world = SimWorld(truth)
    a = SimpleNamespace(out=str(tmp_path))
    proto = js.loads((run.HERE / "protocol.json").read_text())

    def pool(cells, on_done):
        for c in cells:
            on_done(c["key"], world.run_cell(None, None, None, None, proto, c["suite"], c["task"], c["method"], c["condition"], c["episodes"], c["seed"]))

    return world, run.cmd_benchmark(a, proto, pool, {}), proto


def test_benchmark_detects_a_real_gain_and_never_tests_on_dev_tasks(monkeypatch, tmp_path):
    truth = lambda n: 0.85 if n == "cyc_L10_r90" else 0.70 if n == "native_L50" else 0.60
    world, out, proto = run_sim(monkeypatch, tmp_path, truth)
    assert out["ptp"] == "cyc_L10_r90" and out["standard"] == "native_L50"
    assert out["verdict"] == "WIN" and out["primary_ptp_minus_standard"]["diff"] == pytest.approx(0.15, abs=0.04)
    assert out["pairing_valid"] and out["primary_ptp_minus_standard"]["n_episodes"] == 1000
    test_calls = [c for c in world.calls if c[5] == 50]
    assert {(c[1], c[2]) for c in test_calls} == {(s, t) for s, ts in proto["benchmark"]["test"]["tasks"].items() for t in ts}
    assert all(c[0] in {"cyc_L10_r90", "native_L50", "fresh_L10"} for c in test_calls)
    assert {c[4] for c in test_calls} == {1000} and {c[4] for c in world.calls if c[5] == 20} == {2000}


def test_benchmark_does_not_invent_a_win_when_everything_is_equal(monkeypatch, tmp_path):
    _, out, _ = run_sim(monkeypatch, tmp_path, lambda n: 0.7)
    assert out["verdict"] == "NO-IMPROVEMENT-SHOWN" and out["primary_ptp_minus_standard"]["diff"] == 0


def test_benchmark_reports_worse_and_compares_to_the_best_standard(monkeypatch, tmp_path):
    truth = lambda n: 0.9 if n == "fresh_L25" else 0.5 if n in proto_carry else 0.6
    proto_carry = {"shift_L10_r70", "shift_L10_r90", "cyc_L10_r70", "cyc_L10_r90", "same_L10_r90", "shift_L25_r85", "cyc_L25_r85"}
    _, out, _ = run_sim(monkeypatch, tmp_path, truth)
    assert out["standard"] == "fresh_L25" and out["verdict"] == "WORSE-THAN-STANDARD"


def test_results_saved_by_workers_are_not_written_twice_and_survive_restart(tmp_path):
    import run

    path = tmp_path / "cells.jsonl"
    store = run.Store(path)
    # A worker saves the cell itself, then the parent is told. The parent must not append it again.
    run.append_cell(path, dict(key="dev|a", rows=[dict(success=True)], meta={}))
    cell = run.make_cell("dev|a", "s", 0, {}, "clean", 1, 1)

    def pool(cells, on_done):
        assert cells[0]["store"] == str(path), "workers need the file to save into"
        on_done("dev|a", ([dict(success=True)], dict(saved_by_worker=True)))

    run.run_cells([cell], store, pool, "dev")
    assert len(path.read_text().splitlines()) == 1 and "saved_by_worker" not in store.cells["dev|a"]["meta"]
    # Restarted from disk alone, with the parent's memory gone, the finished cell is not run again.
    run.run_cells([cell], run.Store(path), lambda cells, on_done: pytest.fail("finished cell ran again"), "dev")


def test_quick_mode_runs_80_paired_jobs_and_finds_a_known_gain(tmp_path):
    import json as js

    import run

    proto = js.loads((run.HERE / "protocol.json").read_text())
    truth = dict(native_L50=0.5, fresh_L10=0.6, shift_L10_r90=0.6, same_L10_r90=0.8)
    seen = []

    def pool(cells, on_done):
        for c in cells:
            name = c["key"].split("|")[1]
            seen.append(c["key"])
            u = np.random.default_rng(abs(hash((c["suite"], c["task"], c["condition"]))) % 2**32).random(c["episodes"])
            on_done(c["key"], ([dict(suite=c["suite"], task=c["task"], condition=c["condition"], seed=c["seed"], episode=i, success=bool(u[i] < truth[name]), start_hash=str(i)) for i in range(c["episodes"])], dict(seconds=1.0, peak_vram_gb=None)))

    out = run.cmd_quick(SimpleNamespace(out=str(tmp_path)), proto, pool, {})
    assert len(seen) == 80 == len(set(seen))
    assert out["verdict"] == "WIN" and out["primary_same_minus_fresh"]["diff"] == pytest.approx(0.2, abs=0.06)
    assert out["shift_minus_fresh"]["diff"] == 0 and out["primary_same_minus_fresh"]["n_episodes"] == 400
    assert (tmp_path / "quick_results.json").exists()


def test_store_ignores_a_line_cut_off_by_a_crash(tmp_path):
    import run

    path = tmp_path / "cells.jsonl"
    run.append_cell(path, dict(key="dev|a", rows=[], meta={}))
    with open(path, "a") as f:
        f.write('{"key": "dev|b", "rows": [{"succ')
    assert list(run.Store(path).cells) == ["dev|a"]


def test_new_settings_keep_dev_and_test_tasks_separate():
    import json as js

    import run

    b = js.loads((run.HERE / "protocol.json").read_text())["benchmark"]
    assert {"same_L10_r70", "same_L10_r90", "same_L10_r95", "same_L25_r85"} <= set(b["carry"])
    dev = {(s, t) for s, ts in b["dev"]["tasks"].items() for t in ts}
    assert not dev & {(s, t) for s, ts in b["test"]["tasks"].items() for t in ts}
    for m in b["carry"].values():
        assert m["L"] in {x["L"] for x in b["standard"].values()}


def test_benchmark_resumes_without_rerunning_finished_cells(monkeypatch, tmp_path):
    world1, out1, _ = run_sim(monkeypatch, tmp_path, lambda n: 0.8 if n == "shift_L10_r90" else 0.6)
    assert len(world1.calls) > 0
    world2, out2, _ = run_sim(monkeypatch, tmp_path, lambda n: 0.8 if n == "shift_L10_r90" else 0.6)
    assert world2.calls == [] and out2["verdict"] == out1["verdict"]


def test_stale_observation_hook():
    import run

    calls = iter(range(100))
    hook = run.EnvPre(lambda o: o, delay=2)
    seen = []
    for _ in range(5):
        i = next(calls)
        seen.append(hook({"observation.state": torch.full((1, 8), float(i))})["observation.state"][0, 0].item())
    assert seen == [0, 0, 0, 1, 2]
    hook.reset()
    assert hook({"observation.state": torch.full((1, 8), 9.0)})["observation.state"][0, 0].item() == 9.0
    assert len(hook.start_hashes) == 2
