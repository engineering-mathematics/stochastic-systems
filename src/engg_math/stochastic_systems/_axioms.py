"""Probability axioms and Bayes' rule on scalar probabilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._errors import DomainError

if TYPE_CHECKING:
    from collections.abc import Sequence

# Tolerance on sums of probabilities that are exact only up to rounding.
SUM_ATOL = 1e-9


def validate_probability(p: float, name: str = "p") -> float:
    """Return ``p`` as a float, or raise if it is not a finite value in [0, 1]."""
    value = float(p)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DomainError(f"{name} must be a probability in [0, 1], got {p}")
    return value


def complement(p: float) -> float:
    """``P(not A) = 1 - P(A)``."""
    return 1.0 - validate_probability(p)


def union(p_a: float, p_b: float, p_ab: float) -> float:
    """``P(A or B)`` by inclusion-exclusion; ``p_ab`` must be a feasible intersection."""
    a, b, ab = (validate_probability(x, n) for x, n in ((p_a, "p_a"), (p_b, "p_b"), (p_ab, "p_ab")))
    lower, upper = max(0.0, a + b - 1.0), min(a, b)
    if not lower - SUM_ATOL <= ab <= upper + SUM_ATOL:
        raise DomainError(f"p_ab must lie in [{lower}, {upper}] for p_a={a}, p_b={b}, got {ab}")
    return min(1.0, a + b - ab)


def joint_independent(probabilities: Sequence[float]) -> float:
    """Joint probability of independent events: the product."""
    result = 1.0
    for i, p in enumerate(probabilities):
        result *= validate_probability(p, f"probabilities[{i}]")
    return result


def conditional(p_ab: float, p_b: float) -> float:
    """``P(A | B) = P(A and B) / P(B)``; needs ``P(B) > 0`` and ``P(A and B) <= P(B)``."""
    ab, b = validate_probability(p_ab, "p_ab"), validate_probability(p_b, "p_b")
    if b == 0.0:
        raise DomainError("cannot condition on an event of probability zero")
    if ab > b + SUM_ATOL:
        raise DomainError(f"p_ab ({ab}) cannot exceed p_b ({b})")
    return min(1.0, ab / b)


def bayes(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """``P(A | B) = P(B | A) P(A) / P(B)``."""
    likelihood = validate_probability(p_b_given_a, "p_b_given_a")
    prior = validate_probability(p_a, "p_a")
    return conditional(likelihood * prior, p_b)


def _check_partition(priors: Sequence[float], likelihoods: Sequence[float]) -> None:
    if len(priors) != len(likelihoods) or not priors:
        raise DomainError(
            f"priors and likelihoods must be non-empty and equal length, "
            f"got {len(priors)} and {len(likelihoods)}"
        )
    for i, (p, lk) in enumerate(zip(priors, likelihoods, strict=True)):
        validate_probability(p, f"priors[{i}]")
        validate_probability(lk, f"likelihoods[{i}]")
    total = math.fsum(priors)
    if abs(total - 1.0) > SUM_ATOL:
        raise DomainError(f"priors must sum to 1, got {total}")


def total_probability(priors: Sequence[float], likelihoods: Sequence[float]) -> float:
    """``P(B) = sum_i P(B | A_i) P(A_i)`` over a partition ``A_i``."""
    _check_partition(priors, likelihoods)
    return min(1.0, math.fsum(p * lk for p, lk in zip(priors, likelihoods, strict=True)))


def posterior(priors: Sequence[float], likelihoods: Sequence[float]) -> tuple[float, ...]:
    """``P(A_i | B)`` for every cell of the partition; needs ``P(B) > 0``."""
    evidence = total_probability(priors, likelihoods)
    if evidence == 0.0:
        raise DomainError("evidence has probability zero under every hypothesis")
    return tuple(p * lk / evidence for p, lk in zip(priors, likelihoods, strict=True))
