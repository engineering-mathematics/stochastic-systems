"""Exception types."""

from __future__ import annotations

from engg_math.core import EnggMathError


class DomainError(EnggMathError, ValueError):
    """An argument lies outside the domain of the operation."""


class NetworkError(EnggMathError, ValueError):
    """A Bayesian-network definition or query is inconsistent."""
