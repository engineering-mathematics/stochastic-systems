"""Univariate distributions: Normal, Binomial, Poisson."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._axioms import validate_probability
from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]


def _log_pmf_terms(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@dataclass(frozen=True, slots=True)
class Normal:
    """Normal distribution with mean ``mean`` and standard deviation ``std > 0``."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not math.isfinite(self.mean):
            raise DomainError(f"mean must be finite, got {self.mean}")
        if not math.isfinite(self.std) or self.std <= 0.0:
            raise DomainError(f"std must be finite and positive, got {self.std}")

    @property
    def variance(self) -> float:
        """``std**2``."""
        return self.std**2

    def pdf(self, x: float) -> float:
        """Probability density at ``x``."""
        z = (x - self.mean) / self.std
        return math.exp(-0.5 * z * z) / (self.std * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        """``P(X <= x)``."""
        return 0.5 * math.erfc(-(x - self.mean) / (self.std * math.sqrt(2.0)))

    def sample(self, size: int, rng: np.random.Generator | None = None) -> Array:
        """Draw ``size`` variates; pass a seeded ``Generator`` for reproducibility."""
        generator = rng if rng is not None else np.random.default_rng()
        draws: Array = generator.normal(self.mean, self.std, size)
        return draws


@dataclass(frozen=True, slots=True)
class Binomial:
    """Number of successes in ``n`` independent trials with success probability ``p``."""

    n: int
    p: float

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.n < 0:
            raise DomainError(f"n must be a non-negative integer, got {self.n}")
        validate_probability(self.p, "p")

    @property
    def mean(self) -> float:
        """``n p``."""
        return self.n * self.p

    @property
    def variance(self) -> float:
        """``n p (1 - p)``."""
        return self.n * self.p * (1.0 - self.p)

    def pmf(self, k: int) -> float:
        """``P(X = k)``; zero outside ``0..n``."""
        if not 0 <= k <= self.n:
            return 0.0
        if self.p in (0.0, 1.0):
            return float(k == (self.n if self.p == 1.0 else 0))
        log_p = (
            _log_pmf_terms(k, self.n) + k * math.log(self.p) + (self.n - k) * math.log1p(-self.p)
        )
        return math.exp(log_p)

    def cdf(self, k: float) -> float:
        """``P(X <= k)``."""
        if k < 0:
            return 0.0
        upper = min(self.n, math.floor(k))
        return min(1.0, math.fsum(self.pmf(i) for i in range(upper + 1)))

    def sample(self, size: int, rng: np.random.Generator | None = None) -> Array:
        """Draw ``size`` variates."""
        generator = rng if rng is not None else np.random.default_rng()
        draws: Array = generator.binomial(self.n, self.p, size).astype(np.float64)
        return draws


@dataclass(frozen=True, slots=True)
class Poisson:
    """Event count for a process with mean ``rate >= 0`` per interval."""

    rate: float

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not math.isfinite(self.rate) or self.rate < 0.0:
            raise DomainError(f"rate must be finite and non-negative, got {self.rate}")

    @property
    def mean(self) -> float:
        """``rate``."""
        return self.rate

    @property
    def variance(self) -> float:
        """``rate``."""
        return self.rate

    def pmf(self, k: int) -> float:
        """``P(X = k)``; zero for negative ``k``."""
        if k < 0:
            return 0.0
        if self.rate == 0.0:
            return float(k == 0)
        return math.exp(k * math.log(self.rate) - self.rate - math.lgamma(k + 1))

    def cdf(self, k: float) -> float:
        """``P(X <= k)``."""
        if k < 0:
            return 0.0
        return min(1.0, math.fsum(self.pmf(i) for i in range(math.floor(k) + 1)))

    def sample(self, size: int, rng: np.random.Generator | None = None) -> Array:
        """Draw ``size`` variates."""
        generator = rng if rng is not None else np.random.default_rng()
        draws: Array = generator.poisson(self.rate, size).astype(np.float64)
        return draws
