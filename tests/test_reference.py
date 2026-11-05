"""Reference-solution tests against scipy.stats. REQ-SS-003."""

from typing import Any

import pytest
from numpy.testing import assert_allclose
from scipy import stats as _stats

from engg_math.stochastic_systems import Binomial, Normal, Poisson

stats: Any = _stats  # scipy-stubs differ between releases; Any avoids per-version ignores


@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 1.7, 6.0])
def test_normal_matches_scipy(x: float) -> None:
    ours, ref = Normal(1.0, 2.5), stats.norm(1.0, 2.5)
    assert_allclose(ours.pdf(x), ref.pdf(x), rtol=1e-12)
    assert_allclose(ours.cdf(x), ref.cdf(x), rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize(("n", "p"), [(10, 0.3), (200, 0.5), (1000, 0.01)])
def test_binomial_matches_scipy(n: int, p: float) -> None:
    ours, ref = Binomial(n, p), stats.binom(n, p)
    for k in (0, n // 10, n // 2, n):
        assert_allclose(ours.pmf(k), ref.pmf(k), rtol=1e-9, atol=1e-300)
        assert_allclose(ours.cdf(k), ref.cdf(k), rtol=1e-9)


@pytest.mark.parametrize("rate", [0.5, 4.0, 50.0])
def test_poisson_matches_scipy(rate: float) -> None:
    ours, ref = Poisson(rate), stats.poisson(rate)
    for k in (0, 1, int(rate), int(rate) + 10):
        assert_allclose(ours.pmf(k), ref.pmf(k), rtol=1e-9, atol=1e-300)
        assert_allclose(ours.cdf(k), ref.cdf(k), rtol=1e-9)
