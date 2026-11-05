"""Unit and property tests for BayesianNetwork. REQ-SS-004."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.stochastic_systems import BayesianNetwork, DomainError, NetworkError

T, F = True, False


def sprinkler() -> BayesianNetwork:
    net = BayesianNetwork()
    net.add_variable("rain", {(): 0.2})
    net.add_variable("sprinkler", {(F,): 0.4, (T,): 0.01}, parents=["rain"])
    net.add_variable(
        "wet",
        {(F, F): 0.0, (F, T): 0.9, (T, F): 0.8, (T, T): 0.99},
        parents=["rain", "sprinkler"],
    )
    return net


def test_marginal_matches_hand_calculation() -> None:
    net = sprinkler()
    assert_allclose(net.marginal("rain"), 0.2, rtol=1e-12)
    # P(sprinkler) = 0.8*0.4 + 0.2*0.01
    assert_allclose(net.marginal("sprinkler"), 0.322, rtol=1e-12)


def test_explaining_away() -> None:
    net = sprinkler()
    given_wet = net.probability({"rain": T}, {"wet": T})
    given_wet_and_sprinkler = net.probability({"rain": T}, {"wet": T, "sprinkler": T})
    assert given_wet_and_sprinkler < given_wet


def test_conflicting_query_and_evidence_is_impossible() -> None:
    assert sprinkler().probability({"rain": T}, {"rain": F}) == 0.0


def test_structure_queries() -> None:
    net = sprinkler()
    assert net.variables == ("rain", "sprinkler", "wet")
    assert net.parents("wet") == ("rain", "sprinkler")


def test_definition_errors() -> None:
    net = sprinkler()
    with pytest.raises(NetworkError, match="already exists"):
        net.add_variable("rain", {(): 0.5})
    with pytest.raises(NetworkError, match="unknown variable"):
        net.add_variable("x", {(F,): 0.5, (T,): 0.5}, parents=["missing"])
    with pytest.raises(NetworkError, match="duplicate parents"):
        net.add_variable("x", {(F, F): 0.5}, parents=["rain", "rain"])
    with pytest.raises(NetworkError, match="must cover"):
        net.add_variable("x", {(F,): 0.5}, parents=["rain"])
    with pytest.raises(DomainError):
        net.add_variable("x", {(): 1.5})


def test_query_errors() -> None:
    net = sprinkler()
    with pytest.raises(NetworkError, match="unknown variable"):
        net.probability({"nope": T})
    with pytest.raises(NetworkError, match="every variable"):
        net.joint({"rain": T})
    with pytest.raises(DomainError, match="probability zero"):
        net.probability({"rain": T}, {"wet": T, "sprinkler": F, "rain": F})


def test_enumeration_limit() -> None:
    net = BayesianNetwork()
    for i in range(21):
        net.add_variable(f"v{i}", {(): 0.5})
    with pytest.raises(DomainError, match="limit of 20"):
        net.marginal("v0")


@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_joint_sums_to_one_and_conditioning_is_consistent(a: float, b1: float, b0: float) -> None:
    net = BayesianNetwork()
    net.add_variable("a", {(): a})
    net.add_variable("b", {(F,): b0, (T,): b1}, parents=["a"])
    total = sum(net.joint({"a": x, "b": y}) for x in (F, T) for y in (F, T))
    assert_allclose(total, 1.0, atol=1e-12)
    # Bayes: P(a|b) P(b) = P(b|a) P(a)
    p_b = net.marginal("b")
    if p_b > 1e-9:
        lhs = net.probability({"a": T}, {"b": T}) * p_b
        assert_allclose(lhs, b1 * a, atol=1e-9)
