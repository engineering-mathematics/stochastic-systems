"""Unit and property tests for probability axioms. REQ-SS-001."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.stochastic_systems import (
    DomainError,
    bayes,
    complement,
    conditional,
    joint_independent,
    posterior,
    total_probability,
    union,
    validate_probability,
)

prob = st.floats(0.0, 1.0)


def test_known_values() -> None:
    assert complement(0.25) == 0.75
    assert union(0.5, 0.5, 0.25) == 0.75
    assert joint_independent([0.5, 0.5, 0.5]) == 0.125
    assert conditional(0.1, 0.5) == pytest.approx(0.2)


def test_bayes_classic_diagnostic_test() -> None:
    # 1% prevalence, 99% sensitivity, 5% false positives.
    p_positive = total_probability([0.01, 0.99], [0.99, 0.05])
    assert bayes(0.99, 0.01, p_positive) == pytest.approx(0.1667, abs=1e-4)
    assert posterior([0.01, 0.99], [0.99, 0.05])[0] == pytest.approx(0.1667, abs=1e-4)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_rejected(bad: float) -> None:
    with pytest.raises(DomainError):
        validate_probability(bad)


def test_domain_errors() -> None:
    with pytest.raises(DomainError, match="must lie"):
        union(0.5, 0.5, 0.9)
    with pytest.raises(DomainError, match="probability zero"):
        conditional(0.0, 0.0)
    with pytest.raises(DomainError, match="cannot exceed"):
        conditional(0.6, 0.5)
    with pytest.raises(DomainError, match="sum to 1"):
        total_probability([0.5, 0.4], [1.0, 1.0])
    with pytest.raises(DomainError, match="equal length"):
        total_probability([1.0], [0.5, 0.5])
    with pytest.raises(DomainError, match="zero under every"):
        posterior([0.5, 0.5], [0.0, 0.0])


@given(prob)
def test_complement_is_involution(p: float) -> None:
    assert_allclose(complement(complement(p)), p, atol=1e-12)


@given(prob, prob)
def test_union_is_bounded_by_sum_and_max(a: float, b: float) -> None:
    ab = a * b  # independent intersection is always feasible
    u = union(a, b, ab)
    assert max(a, b) - 1e-12 <= u <= min(1.0, a + b) + 1e-12


@given(st.lists(st.floats(0.01, 1.0), min_size=1, max_size=6))
def test_posterior_sums_to_one(weights: list[float]) -> None:
    total = sum(weights)
    priors = [w / total for w in weights]
    post = posterior(priors, [1.0] * len(priors))
    assert_allclose(sum(post), 1.0, atol=1e-9)
    assert_allclose(post, priors, atol=1e-9)
