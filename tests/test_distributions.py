"""Unit and property tests for distributions. REQ-SS-002."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.stochastic_systems import Binomial, DomainError, Normal, Poisson


def test_parameter_validation() -> None:
    with pytest.raises(DomainError):
        Normal(0.0, 0.0)
    with pytest.raises(DomainError):
        Normal(float("nan"), 1.0)
    with pytest.raises(DomainError):
        Binomial(-1, 0.5)
    with pytest.raises(DomainError):
        Binomial(3, 1.5)
    with pytest.raises(DomainError):
        Poisson(-1.0)


def test_normal_standard_values() -> None:
    n = Normal(0.0, 1.0)
    assert_allclose(n.pdf(0.0), 1 / math.sqrt(2 * math.pi), rtol=1e-12)
    assert_allclose(n.cdf(0.0), 0.5, rtol=1e-12)
    assert_allclose(n.cdf(1.96), 0.975, atol=1e-4)
    assert n.variance == 1.0


def test_binomial_known_values_and_support() -> None:
    b = Binomial(4, 0.5)
    assert_allclose(b.pmf(2), 0.375, rtol=1e-12)
    assert b.pmf(5) == 0.0
    assert b.cdf(-1) == 0.0
    assert_allclose(b.cdf(4), 1.0, rtol=1e-12)
    assert (b.mean, b.variance) == (2.0, 1.0)


def test_binomial_degenerate_probabilities() -> None:
    assert Binomial(5, 0.0).pmf(0) == 1.0
    assert Binomial(5, 0.0).pmf(2) == 0.0
    assert Binomial(5, 1.0).pmf(5) == 1.0
    assert Binomial(5, 1.0).pmf(4) == 0.0


def test_poisson_known_values_and_degenerate_rate() -> None:
    p = Poisson(2.0)
    assert_allclose(p.pmf(0), math.exp(-2.0), rtol=1e-12)
    assert p.pmf(-1) == 0.0
    assert p.cdf(-1) == 0.0
    assert (p.mean, p.variance) == (2.0, 2.0)
    assert Poisson(0.0).pmf(0) == 1.0
    assert Poisson(0.0).pmf(3) == 0.0


def test_sampling_is_reproducible_and_matches_moments() -> None:
    for dist in (Normal(3.0, 2.0), Binomial(20, 0.3), Poisson(4.0)):
        a = dist.sample(20000, np.random.default_rng(7))
        b = dist.sample(20000, np.random.default_rng(7))
        assert_allclose(a, b)
        assert_allclose(a.mean(), dist.mean, atol=0.1)


def test_sampling_without_generator_returns_requested_size() -> None:
    assert Normal(0.0, 1.0).sample(5).shape == (5,)
    assert Binomial(3, 0.5).sample(5).shape == (5,)
    assert Poisson(1.0).sample(5).shape == (5,)


@given(st.integers(0, 60), st.floats(0.0, 1.0))
def test_binomial_pmf_sums_to_one(n: int, p: float) -> None:
    dist = Binomial(n, p)
    assert_allclose(math.fsum(dist.pmf(k) for k in range(n + 1)), 1.0, atol=1e-9)


@given(st.floats(0.0, 30.0))
def test_poisson_cdf_approaches_one(rate: float) -> None:
    assert_allclose(Poisson(rate).cdf(rate + 12 * math.sqrt(rate + 1) + 20), 1.0, atol=1e-9)


@given(st.floats(-10, 10), st.floats(0.1, 10), st.floats(-50, 50), st.floats(-50, 50))
def test_normal_cdf_is_monotone(mean: float, std: float, x: float, y: float) -> None:
    lo, hi = sorted((x, y))
    d = Normal(mean, std)
    assert d.cdf(lo) <= d.cdf(hi)
