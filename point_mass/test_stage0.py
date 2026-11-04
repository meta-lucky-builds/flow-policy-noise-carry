"""Validates the Stage 0 instruments on a case with a known answer: an exact two-mode FM flow."""
import numpy as np
import pytest
import torch

import stage0

H, D = 16, 2


class ExactTwoMode:
    """Exact velocity for A ~ 0.5 N(+m, s^2 I) + 0.5 N(-m, s^2 I). Ignores observation."""

    def __init__(self, sep=1.0, s=0.1):
        self.m = torch.ones(H, D) * sep
        self.s2 = s ** 2

    def solve(self, z, obs, steps=10):
        x = z
        n = 40
        for i in range(n):
            t = i / n
            c = (1 - t) ** 2 + t ** 2 * self.s2
            ra, rb = x - t * self.m, x + t * self.m
            la, lb = -(ra ** 2).sum((1, 2)) / (2 * c), -(rb ** 2).sum((1, 2)) / (2 * c)
            wa = torch.sigmoid(la - lb)[:, None, None]
            g = (t * self.s2 - (1 - t)) / c
            x = x + (wa * (self.m + g * ra) + (1 - wa) * (-self.m + g * rb)) / n
        return x

    def adapter(self, x):
        return x.clamp(-1, 1) * 2


def test_two_means_does_not_flag_a_single_gaussian_but_flags_two_modes():
    rng = np.random.default_rng(0)
    gauss = [stage0.two_means_1d(rng.standard_normal(128))[2] for _ in range(200)]
    assert np.percentile(gauss, 99) < 3.5
    lab = rng.random(128) < 0.5
    two = np.where(lab, 1.0, -1.0) + 0.1 * rng.standard_normal(128)
    assert stage0.two_means_1d(two)[2] > 3.5


def test_screen_finds_modes_on_exact_flow():
    gen = torch.Generator().manual_seed(0)
    s = stage0.screen_state(ExactTwoMode(), np.zeros(10), 256, gen)
    assert s["ashman"] > 3.5 and s["minor"] > 0.3


def test_keep_rate_matches_known_answers():
    """Uniform separation: shift keeps 1/2+arcsin(.75 rho)/pi, same-index keeps 1/2+arcsin(rho)/pi."""
    gen = torch.Generator().manual_seed(1)
    p = ExactTwoMode()
    states = []
    for _ in range(4):
        states.append(dict(ob=np.zeros(10), screen=stage0.screen_state(p, np.zeros(10), 1000, gen)))
    k = stage0.keep_rates(p, states, 1000, gen, [0.0, 0.95], ["shift", "same"])
    assert k["shift"]["0.0"] == pytest.approx(0.5, abs=0.04)
    assert k["same"]["0.0"] == pytest.approx(0.5, abs=0.04)
    assert k["shift"]["0.95"] == pytest.approx(0.5 + np.arcsin(0.75 * 0.95) / np.pi, abs=0.04)
    assert k["same"]["0.95"] == pytest.approx(0.5 + np.arcsin(0.95) / np.pi, abs=0.04)
