# Testing

Aragog uses [pytest](https://docs.pytest.org/) for testing.

## Prerequisites

Install the test dependencies:

```sh
pip install -e ".[docs]"
pip install pytest pytest-cov pytest-dependency
```

Or with Poetry:

```sh
poetry install --with test
```

## Running tests

```sh
# All tests
pytest tests/

# A specific file
pytest tests/test_entropy_pytest.py
pytest tests/test_entropy_verification.py
pytest tests/test_jax_entropy.py

# Marker subsets
pytest -m unit tests/
pytest -m smoke tests/

# Recommended: thread-based timeout for sandbox compatibility
pytest -p no:faulthandler --timeout=60 --timeout-method=thread tests/
```

The JAX-path tests require `jax` and `equinox`; tests are skipped (not failed) when those optional dependencies are missing. Some tests also require the SPIDER-format EOS tables; they skip if `FWL_DATA` is unset or the expected files are not present.

## Test suite overview

The current test suite is organised by the layer it covers:

- **`tests/test_entropy_pytest.py`** — EOS unit tests, phase evaluator tests, solver-state and IC tests, mesh tests. The bulk of the unit-level coverage.
- **`tests/test_entropy_verification.py`** — Conservation laws (energy, mass) and grey-body cooling against analytic limits.
- **`tests/test_entropy_advanced.py`** — Advanced solver tests (extended-state modes, retry-ladder hooks).
- **`tests/test_jax_entropy.py`** — JAX-path parity: EOS, phase evaluator, JIT/`vmap`/`grad`, boundary copies, solver parity against the numpy path.
- **`tests/test_jax_mesh_gravity_fallback.py`** — External-mesh per-node gravity profile is interpolated correctly when `eos_method = 2`.

## What the tests verify

1. **Solver completion.** The integrator reaches the requested end time and returns `status = 0` for representative configurations.
2. **Physical plausibility.** Temperatures stay positive, the smooth-clipped melt fraction stays in $[0, 1]$, fluxes have the correct sign relative to the entropy gradient, and the rheological-front depth is non-negative.
3. **EOS correctness.** Property lookups at on- and off-grid $(P, S)$ points return finite, monotone values; the SPIDER-parity two-stage blend is continuous across the solidus and liquidus.
4. **Conservation.** Energy and mass conservation under closed integrations to within solver tolerance; mass conservation also holds through gravitational separation and chemical mixing (the segregation flux is divergence-free in mass).
5. **JAX/numpy parity.** `EntropyEOS_JAX` and `compute_fluxes` reproduce the numpy path's outputs to within floating-point tolerance, and a CVODE run with the JAX RHS produces a trajectory that matches the numpy RHS run.

## Coverage

```sh
pytest --cov=src/aragog --cov-report=html tests/
```

Open `htmlcov/index.html` to inspect line-by-line coverage.

## Linting

Before committing, format and lint:

```sh
ruff check --fix src/ tests/
ruff format src/ tests/
```
