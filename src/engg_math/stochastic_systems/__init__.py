"""Probability axioms, distributions and Bayesian-network inference."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._axioms import (
    bayes,
    complement,
    conditional,
    joint_independent,
    posterior,
    total_probability,
    union,
    validate_probability,
)
from ._distributions import Binomial, Normal, Poisson
from ._errors import DomainError, NetworkError
from ._network import BayesianNetwork

try:
    __version__ = version("engg-math-stochastic-systems")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "BayesianNetwork",
    "Binomial",
    "DomainError",
    "NetworkError",
    "Normal",
    "Poisson",
    "__version__",
    "bayes",
    "complement",
    "conditional",
    "joint_independent",
    "posterior",
    "total_probability",
    "union",
    "validate_probability",
]
