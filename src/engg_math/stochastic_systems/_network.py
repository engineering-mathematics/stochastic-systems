"""Discrete Bayesian network over binary variables with exact inference by enumeration."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from engg_math.discrete_structures import DirectedGraph

from ._axioms import validate_probability
from ._errors import DomainError, NetworkError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Inference is exponential in the number of unobserved variables.
MAX_FREE_VARIABLES = 20


class BayesianNetwork:
    """Directed acyclic network of binary variables.

    Variables are added after their parents, so the graph is acyclic by construction.
    Conditional tables map each tuple of parent values (in declared order) to ``P(variable=True)``.
    """

    def __init__(self) -> None:
        """Create an empty network."""
        self._graph = DirectedGraph()
        self._parents: dict[str, tuple[str, ...]] = {}
        self._tables: dict[str, dict[tuple[bool, ...], float]] = {}

    @property
    def variables(self) -> tuple[str, ...]:
        """Variable names in insertion order."""
        return tuple(self._parents)

    def parents(self, name: str) -> tuple[str, ...]:
        """Return the declared parents of ``name``."""
        self._require(name)
        return self._parents[name]

    def _require(self, name: str) -> None:
        if name not in self._parents:
            raise NetworkError(f"unknown variable {name!r}")

    def add_variable(
        self,
        name: str,
        table: Mapping[tuple[bool, ...], float],
        parents: Sequence[str] = (),
    ) -> None:
        """Add ``name`` with ``P(name=True | parents)`` given for every parent assignment."""
        if name in self._parents:
            raise NetworkError(f"variable {name!r} already exists")
        parent_names = tuple(parents)
        if len(set(parent_names)) != len(parent_names):
            raise NetworkError(f"duplicate parents for {name!r}: {parent_names}")
        for parent in parent_names:
            self._require(parent)
        expected = set(product((False, True), repeat=len(parent_names)))
        if set(table) != expected:
            raise NetworkError(
                f"table for {name!r} must cover exactly the {len(expected)} parent assignments"
            )
        checked = {key: validate_probability(p, f"table[{key}]") for key, p in table.items()}
        self._parents[name] = parent_names
        self._tables[name] = checked
        self._graph.add_node(name)
        for parent in parent_names:
            self._graph.add_edge(parent, name)

    def joint(self, assignment: Mapping[str, bool]) -> float:
        """Probability of a complete assignment: the product of each variable's conditional."""
        if set(assignment) != set(self._parents):
            raise NetworkError("joint needs a value for every variable and no others")
        result = 1.0
        for name, parents in self._parents.items():
            p_true = self._tables[name][tuple(assignment[p] for p in parents)]
            result *= p_true if assignment[name] else 1.0 - p_true
        return result

    def _mass(self, fixed: Mapping[str, bool]) -> float:
        free = [str(v) for v in self._graph.topological_order() if v not in fixed]
        if len(free) > MAX_FREE_VARIABLES:
            raise DomainError(
                f"{len(free)} unobserved variables exceed the limit of {MAX_FREE_VARIABLES}"
            )
        total = 0.0
        for values in product((False, True), repeat=len(free)):
            total += self.joint({**fixed, **dict(zip(free, values, strict=True))})
        return total

    def probability(
        self, query: Mapping[str, bool], evidence: Mapping[str, bool] | None = None
    ) -> float:
        """``P(query | evidence)`` by enumeration; evidence must have nonzero probability."""
        given = dict(evidence or {})
        for name in (*query, *given):
            self._require(name)
        denominator = self._mass(given)
        if denominator == 0.0:
            raise DomainError("evidence has probability zero")
        if any(name in given and given[name] != value for name, value in query.items()):
            return 0.0
        return min(1.0, self._mass({**given, **query}) / denominator)

    def marginal(self, name: str, evidence: Mapping[str, bool] | None = None) -> float:
        """``P(name=True | evidence)``."""
        return self.probability({name: True}, evidence)
