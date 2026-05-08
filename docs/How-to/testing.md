# Testing suite

[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/FormingWorlds/aragog/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/aragog)

This page is about *running* the existing test suite. For guidance on *writing*
new tests see [How to build tests](build_tests.md).

Aragog uses [pytest](https://docs.pytest.org/) with
[pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution.
Tests are categorised by speed and purpose into three pytest markers.

## Prerequisites

Install the test extras:

```console
pip install -e ".[test]"
```

The JAX-path tests additionally need the `jax` extra (`pip install -e ".[jax]"`); without it the JAX parity tests are skipped, not failed. SPIDER-format EOS tables are also required for a subset of tests; if `FWL_DATA` is unset or the expected files are missing, those tests skip cleanly.

## Markers

| Marker | Tests | Wall | Scope |
|---|---|---|---|
| `unit` | ~366 | ~1 to 2 min | EOS lookups, mesh helpers, phase evaluator branches, parser validation, JAX-vs-numpy parity on point inputs, regression pins on permeability constants, energy-equation invariants. No real solver call beyond a handful of cheap analytic-EOS smoke checks. |
| `smoke` | ~40 | ~5 to 15 min | Full `EntropySolver.solve()` runs at relaxed tolerance. Verify the whole code path under representative configurations (closed mantle, gravitational separation, JAX RHS via CVODE). |
| `slow` | ~3 | ~30+ min each | Long multi-Myr coupled-style runs and convergence studies. Manual only. |

`pytest --collect-only -m <marker>` reports the live count.

## Running tests

### By marker

```console
pytest -m unit                       # Fast feedback during development
pytest -m smoke                      # Full-solver smoke
pytest -m "unit or smoke"            # CI push tier
pytest -m "unit or smoke or slow"    # Full nightly tier
```

### Single test

```console
pytest tests/test_entropy_pytest.py::TestEnergyBalanceCoreBC::test_energy_balance_rhs_bit_parity_prescribed_inputs
```

### Parallel runs

`pyproject.toml` does not set a default `addopts`; `pytest` runs serial unless `-n auto` (or another xdist option) is supplied explicitly. CI invokes `pytest -m "unit and not slow" -n auto` (`ci_tests.yml`) and `pytest -m "unit or smoke or slow" -n auto` (`nightly.yml`); reproduce locally by adding the same flag:

```console
pytest -m unit -n auto                       # parallel unit run
pytest -m "unit or smoke" -n auto -ra -v     # parallel + summary + verbose
```

Drop `-n auto` for serial execution when debugging a flaky test or attaching a debugger.

### Sandbox-friendly invocation

Some environments forbid `signal`-based timeouts. Use a thread-based timeout instead:

```console
pytest -p no:faulthandler --timeout=60 --timeout-method=thread tests/
```

## CI tiers

| Trigger | Markers | Budget | Coverage |
|---|---|---|---|
| Push / PR (`ci_tests.yml`) | `unit and not slow` | < 5 min | None |
| Nightly cron (`nightly.yml`, 02:30 UTC) | `unit or smoke or slow` | < 75 min | Yes; uploaded to Codecov |
| Manual `workflow_dispatch` | as above | < 75 min | Yes |

Push CI is intentionally unit-only because each smoke test runs a full `EntropySolver` call (5 to 15 min on a 2-vCPU runner under coverage instrumentation). Burning that budget on every push gives no bug-finding signal that the unit tier doesn't already cover.

## Fixtures

Shared fixtures live in `tests/conftest.py`. The most load-bearing one is `shared_eos`:

### `shared_eos` (session)

A session-scoped EOS loader that opens the SPIDER-format pressure-entropy tables once per test session and hands the `EntropyEOS` instance to every test that needs it. Without this fixture, the integration tests would each rebuild the lookup tables (~ 12 to 15 s per test); with it, the cost amortises to a single load (~ 3.5 s) across the whole nightly run.

If `FWL_DATA` is unset or the expected files are missing, `shared_eos` skips the dependent tests rather than failing.

## Parallelization

Tests are written to be order-independent and run cleanly under `pytest-xdist`. Pass `-n auto` to use all available cores; CI does this on both the unit and nightly tiers. If you observe flakiness only under xdist, that is a bug in the test (not in xdist).

## Coverage

```console
pytest --cov=src/aragog --cov-report=html -m "unit or smoke"
```

Open `htmlcov/index.html` to inspect line-by-line coverage. The nightly CI
uses `--cov-report=xml` and uploads the result to Codecov; the project floor
is 85% (CI gate enforced via `pyproject.toml`).

## Linting

Before committing, format and check all files:

```console
ruff check --fix src/ tests/
ruff format src/ tests/
```

The local ruff (often 0.12.x) and the CI ruff (0.15.x) sometimes disagree on
formatting drift; CI is canonical. Run BOTH `ruff check` and `ruff format`
before pushing — `format` does NOT catch lint rules like `E402` misplaced
imports.
