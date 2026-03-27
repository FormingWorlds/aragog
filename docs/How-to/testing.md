# Testing

Aragog uses [pytest](https://docs.pytest.org/) for testing.

## Prerequisites

Install the test dependencies:

```console
pip install -e ".[docs]"
pip install pytest pytest-cov pytest-dependency
```

Or if using Poetry:

```console
poetry install --with test
```

## Running tests

### All tests

```console
pytest
```

### By file

```console
pytest tests/test_abe.py
pytest tests/test_phase.py
```

## Test suite overview

The test suite includes:

- **`test_abe.py`**: End-to-end solver tests for different configuration files (solid, liquid, mixed phase, mixed with lookup tables, mixed with initial condition). Each test loads a `.cfg` configuration, runs the full solver, and verifies convergence and output ranges.
- **`test_phase.py`**: Unit tests for the phase evaluator system (single phase, mixed phase, composite phase evaluator logic).

## What the tests verify

1. **Solver convergence**: the BDF integrator reaches the requested end time without failure.
2. **Physical plausibility**: temperatures remain positive, melt fractions stay in [0, 1], fluxes have the correct sign.
3. **Phase evaluator correctness**: property lookups (density, viscosity, heat capacity, thermal expansivity) return sensible values in each phase regime.
4. **Configuration loading**: both TOML and INI configurations parse correctly and produce valid `Parameters` objects.

## Coverage

Generate a coverage report:

```console
pytest --cov=src/aragog --cov-report=html
```

Open `htmlcov/index.html` to inspect line-by-line coverage.

## Linting

Before committing, format and lint:

```console
ruff check --fix src/ tests/
ruff format src/ tests/
```
