"""Applicable spec section 17 checks for the K=1 FM path. The model is untrained: these test plumbing and law."""
import numpy as np
import pytest
import torch
from scipy.stats import norm

import env
import expert
from model import FMPolicy
from train import Windows
from wrapper import (ExecutionLedger, LedgerError, carry_source, commitment_kl, latent_logprob, propose)

H, D, L = 16, 2, 4


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    p = FMPolicy(width=32, layers=1, heads=2, ff=64).eval()
    p.version = "v1"
    return p


def obs(n=1):
    return torch.zeros(n, env.OBS_DIM)


def g(s=0):
    return torch.Generator().manual_seed(s)


def test_rho0_reproduces_native_under_same_noise(policy):
    z = torch.randn(4, H, D, generator=g(1))
    eps = torch.randn(4, H, D, generator=g(7))
    out = carry_source(z, 0.0, L, g(7))
    assert torch.equal(out, eps)
    assert torch.equal(policy.decode(out, obs(4)), policy.decode(eps, obs(4)))


def test_supplied_noise_is_consumed_and_decoder_deterministic(policy):
    z = torch.randn(1, H, D, generator=g(2))
    assert torch.equal(policy.decode(z, obs()), policy.decode(z, obs()))
    assert not torch.equal(policy.decode(z, obs()), policy.decode(z + 1, obs()))


@pytest.mark.parametrize("rho", [0.25, 0.7, 0.95])
def test_conditional_law_mean_variance_and_tail(rho):
    zprev = torch.randn(1, H, D, generator=g(3)).repeat(200000, 1, 1)
    z = carry_source(zprev, rho, L, g(4))
    m, v = z.mean(0), z.var(0)
    assert torch.allclose(m[:H - L], rho * zprev[0, L:], atol=0.012)
    assert torch.allclose(v[:H - L], torch.full_like(v[:H - L], 1 - rho ** 2), atol=0.012)
    assert torch.allclose(m[H - L:], torch.zeros(L, D), atol=0.012)
    assert torch.allclose(v[H - L:], torch.ones(L, D), atol=0.012)


def test_rho_cap_enforced():
    with pytest.raises(ValueError):
        carry_source(torch.zeros(1, H, D), 0.96, L, g())


def test_reset_gives_standard_normal():
    zprev = torch.ones(100000, H, D) * 3
    z = carry_source(zprev, 0.95, L, g(5), valid=torch.zeros(100000, dtype=torch.bool))
    assert z.mean().abs() < 0.01 and (z.var() - 1).abs() < 0.01


def test_unconditional_marginal_stays_standard_normal_and_iid():
    z = torch.randn(100000, H, D, generator=g(6))
    for _ in range(6):
        z = carry_source(z, 0.95, L, g(_ + 10))
    f = z.reshape(len(z), -1)
    C = torch.cov(f.T)
    assert torch.allclose(C, torch.eye(H * D), atol=0.03)


def test_memory_flushes_exactly_after_H_over_L_replans():
    z0 = torch.randn(200000, H, D, generator=g(8))
    z = z0
    for k in range(4):
        z = carry_source(z, 0.95, L, g(20 + k))
    cross = (z.reshape(len(z), -1).T @ z0.reshape(len(z), -1)) / len(z)
    assert cross.abs().max() < 0.02
    z3 = z0
    for k in range(3):
        z3 = carry_source(z3, 0.95, L, g(30 + k))
    expected = 0.95 ** 3
    assert abs(float((z3[:, 0, 0] * z0[:, 12, 0]).mean()) - expected) < 0.02


def test_logprob_matches_analytic():
    z = torch.randn(H, D, generator=g(9))
    zb = torch.randn(H, D, generator=g(10))
    mask = torch.cat([torch.ones(H - L), torch.zeros(L)])
    rho = 0.7
    r = (rho * mask)[:, None].numpy()
    ref = norm.logpdf(z.numpy(), loc=r * zb.numpy(), scale=np.sqrt(1 - r ** 2)).sum()
    assert abs(latent_logprob(z, zb, mask, rho) - ref) < 1e-3


def test_kl_formula_matches_monte_carlo():
    zb = torch.randn(H, D, generator=g(11), dtype=torch.float64)
    mask = torch.cat([torch.ones(H - L), torch.zeros(L)]).double()
    rho = 0.95
    r = (rho * mask)[:, None]
    z = r * zb + torch.sqrt(1 - r ** 2) * torch.randn(200000, H, D, generator=g(12), dtype=torch.float64)
    logq = (-0.5 * ((z - r * zb) ** 2 / (1 - r ** 2) + torch.log(2 * np.pi * (1 - r ** 2)))).sum((1, 2))
    logp = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum((1, 2))
    mc = (logq - logp)
    assert abs(float(mc.mean()) - commitment_kl(zb, mask, rho)) < 4 * float(mc.std()) / np.sqrt(len(mc))


def test_anchor_is_input_noise_not_denoised_actions(policy):
    led = ExecutionLedger("v1")
    pid, act, _ = propose(policy, np.zeros(env.OBS_DIM), led, 0.0, g(13))
    stored = led.prop.z.clone()
    assert not torch.allclose(stored, policy.solve(stored[None], obs())[0])
    assert not torch.allclose(stored, act)
    for i in range(L):
        led.ack(led.dispatch(pid, i))
    zb, mask = led.anchor()
    assert torch.equal(zb[:H - L], stored[L:]) and mask[:H - L].all() and not mask[H - L:].any()


def test_ledger_path_matches_batch_law(policy):
    led = ExecutionLedger("v1")
    pid, _, _ = propose(policy, np.zeros(env.OBS_DIM), led, 0.0, g(14))
    for i in range(L):
        led.ack(led.dispatch(pid, i))
    z0 = led.prop.z.clone()
    propose(policy, np.zeros(env.OBS_DIM), led, 0.85, g(15))
    eps = torch.randn(H, D, generator=g(15))
    want = torch.zeros(H, D)
    want[:H - L] = 0.85 * z0[L:] + np.sqrt(1 - 0.85 ** 2) * eps[:H - L]
    want[H - L:] = eps[H - L:]
    assert torch.allclose(led.prop.z, want, atol=1e-6)


def _ledger():
    led = ExecutionLedger("v1")
    pid = led.register(torch.randn(H, D, generator=g(16)))
    return led, pid


def test_partial_consumption_reindexes_by_acknowledged_count():
    led, pid = _ledger()
    z = led.prop.z.clone()
    for i in range(3):
        led.ack(led.dispatch(pid, i))
    zb, mask = led.anchor()
    assert torch.equal(zb[:H - 3], z[3:]) and int(mask.sum()) == H - 3


def test_no_token_executes_twice():
    led, pid = _ledger()
    led.dispatch(pid, 0)
    with pytest.raises(LedgerError):
        led.dispatch(pid, 0)


def test_stale_proposal_cannot_dispatch():
    led, pid = _ledger()
    led.register(torch.zeros(H, D))
    with pytest.raises(LedgerError):
        led.dispatch(pid, 0)


@pytest.mark.parametrize("scenario,reason", [
    ("duplicate", "duplicate_ack"), ("out_of_order", "out_of_order_ack"), ("skipped", "skipped_token"),
    ("cancelled", "token_cancelled"), ("modified", "token_modified"), ("missing_ack", "unacknowledged_tokens"),
    ("unknown", "unknown_ack")])
def test_ambiguous_alignment_discards_anchor(scenario, reason):
    led, pid = _ledger()
    if scenario == "duplicate":
        d = led.dispatch(pid, 0); led.ack(d); led.ack(d)
    elif scenario == "out_of_order":
        a = led.dispatch(pid, 0); b = led.dispatch(pid, 1); led.ack(b); led.ack(a)
    elif scenario == "skipped":
        led.ack(led.dispatch(pid, 1))
    elif scenario == "cancelled":
        led.ack(led.dispatch(pid, 0), "cancelled")
    elif scenario == "modified":
        led.ack(led.dispatch(pid, 0), "modified")
    elif scenario == "missing_ack":
        led.dispatch(pid, 0)
    elif scenario == "unknown":
        led.ack(999)
    zb, mask = led.anchor()
    assert mask.sum() == 0 and zb.abs().sum() == 0 and led.reset_reason == reason


def test_version_change_discards_anchor():
    led, pid = _ledger()
    led.ack(led.dispatch(pid, 0))
    zb, mask = led.anchor("v2")
    assert mask.sum() == 0 and led.reset_reason == "version_change"


def test_observation_contains_no_action_history():
    assert env.OBS_DIM == 10
    assert not any(("act" in f or "cmd" in f or "u_" in f) for f in env.OBS_FIELDS)


def test_memory_causality_future_demos_cannot_change_inputs(policy):
    """Changing future demonstrated actions leaves window observations and every proposal unchanged."""
    o = np.random.default_rng(0).normal(size=(40, env.OBS_DIM)).astype(np.float32)
    a = np.random.default_rng(1).normal(size=(40, 2)).astype(np.float32)
    a2 = a.copy(); a2[10:] += 5
    w1, w2 = Windows(o, a, np.array([40]), H, -np.ones(2), np.ones(2)), Windows(o, a2, np.array([40]), H, -np.ones(2), np.ones(2))
    ob1, act1, _ = w1.batch(np.array([3])); ob2, act2, _ = w2.batch(np.array([3]))
    assert torch.equal(ob1, ob2) and not torch.equal(act1, act2)
    l1, l2 = ExecutionLedger("v1"), ExecutionLedger("v1")
    _, p1, _ = propose(policy, ob1[0].numpy(), l1, 0.5, g(17))
    _, p2, _ = propose(policy, ob2[0].numpy(), l2, 0.5, g(17))
    assert torch.equal(p1, p2)


def test_windows_mask_never_crosses_segment():
    o = np.zeros((30, env.OBS_DIM), np.float32)
    a = np.arange(60, dtype=np.float32).reshape(30, 2)
    w = Windows(o, a, np.array([10, 20]), H, np.array([0., 0.]), np.array([59., 59.]))
    _, act, mask = w.batch(np.array([5]))
    assert mask[0].sum() == 5
    assert torch.allclose(act[0, 5:], act[0, 4].expand(11, 2))


def test_dynamics_and_swept_collision():
    x, v = env.step(np.array([0., 0.]), np.array([1., 0.]), np.array([100., 0.]))
    assert np.allclose(x, [0.05, 0.]) and np.allclose(v, [1., 0.])
    assert env.seg_point_dist(np.array([-1., 0.]), np.array([1., 0.]), np.array([0., 0.1])) == pytest.approx(0.1)


def test_expert_trajectories_are_verified_on_both_routes():
    rng = np.random.default_rng(3)
    sc = env.sample_scene(rng)
    seen = {}
    for r in (1, -1):
        out = expert.plan(sc["start"], np.zeros(2), sc["goal"], sc["obst"], r, 100)
        assert out is not None
        U, xs, vs = out
        i = np.argmin(np.abs(xs[:, 0] - sc["obst"][0]))
        seen[r] = np.sign(xs[i, 1] - sc["obst"][1])
        assert np.abs(U).max() <= env.U_MAX + 1e-9
        assert np.abs(xs[-1] - sc["goal"]).max() <= 0.05 + 1e-4
        assert env.seg_point_dist(xs[:-1], xs[1:], sc["obst"]).min() >= env.OBST_R
    assert seen[1] == 1 and seen[-1] == -1


def test_verifier_rejects_collision():
    sc = dict(start=np.array([-0.3, 0.]), goal=np.array([0.3, 0.]), obst=np.zeros(2))
    U = np.tile([2., 0.], (40, 1))
    ok, _, _ = expert.verify(sc["start"], np.zeros(2), sc["goal"], sc["obst"], U)
    assert not ok
