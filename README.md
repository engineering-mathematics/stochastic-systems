# engg-math-stochastic-systems

Probability axioms, common distributions and exact Bayesian-network inference for `engg_math`.
Every probability is validated to lie in [0, 1]; invalid input raises, it is never clipped.

```bash
pip install engg-math-stochastic-systems
```

```python
from engg_math.stochastic_systems import Binomial, BayesianNetwork, posterior

Binomial(10, 0.3).cdf(4)  # P(X <= 4)
posterior([0.01, 0.99], [0.99, 0.05])  # diagnostic-test posteriors

net = BayesianNetwork()
net.add_variable("rain", {(): 0.2})
net.add_variable("wet", {(False,): 0.1, (True,): 0.9}, parents=["rain"])
net.probability({"rain": True}, {"wet": True})
```

## API

| Area | Names |
|---|---|
| Axioms | `validate_probability`, `complement`, `union`, `joint_independent`, `conditional`, `bayes`, `total_probability`, `posterior` |
| Distributions | `Normal`, `Binomial`, `Poisson`: `pmf`/`pdf`, `cdf`, `mean`, `variance`, `sample` |
| Networks | `BayesianNetwork`: `add_variable`, `joint`, `probability`, `marginal` |
| Errors | `DomainError`, `NetworkError` (subclass `EnggMathError`) |

## Notes

- PMFs use log-gamma terms, so large `n` and `k` do not overflow.
- Network variables are binary. Inference enumerates the unobserved variables (limit 20); it is exact, not scalable.
- Sampling takes a `numpy.random.Generator`; pass a seeded one for reproducible draws.
- The legacy `traverse_edge_intersection` is replaced by `BayesianNetwork.probability`.

## Development

```bash
uv sync
uv run pytest && uv run ruff check && uv run mypy && uv run lint-imports
```

Licensed under Apache-2.0.
