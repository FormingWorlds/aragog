# Testing suite

[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/aragog?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/aragog)

This page is about *running* the existing test suite. For guidance on *writing*
new tests see [How to build tests](build_tests.md).

Aragog uses [pytest](https://docs.pytest.org/) with
[pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution.
Tests are categorised by speed and purpose into four pytest markers (`unit`,
`smoke`, `integration`, `slow`); the marker set is the canonical authority and
is registered under `[tool.pytest.ini_options].markers` in `pyproject.toml`
with `--strict-markers` enforced.

## Prerequisites

Install the test extras:

```console
pip install -e ".[test]"
```

The JAX-path tests additionally need the `jax` extra (`pip install -e ".[jax]"`); without it the JAX parity tests are skipped, not failed. SPIDER-format EOS tables are also required for a subset of tests; if `FWL_DATA` is unset or the expected files are missing, those tests skip cleanly.

## Markers

| Marker | Tests | Wall | Scope |
|---|---|---|---|
| `unit` | ~514 | ~2 min (xdist) | EOS lookups, mesh helpers, phase evaluator branches, parser validation, JAX-vs-numpy parity on point inputs, regression pins on permeability constants, energy-equation invariants. No real solver call beyond a handful of cheap analytic-EOS smoke checks. |
| `smoke` | ~62 | ~10 min (xdist) | Full `EntropySolver.solve()` runs at relaxed tolerance. Verify the whole code path under representative configurations (closed mantle, gravitational separation, JAX RHS via CVODE). |
| `integration` | 0 today | n/a | Real-physics integration against published references (PALEOS, SPIDER bit-parity). Reserved for tests added under this marker; nightly picks them up automatically once present. |
| `slow` | ~3 | ~30+ min each | Long multi-Myr coupled-style runs and convergence studies. Manual only. |

`pytest --collect-only -m <marker>` reports the live count. Total across the
non-skip filter (`-m "not skip"`) on `tl/interior-refactor`: ~575.

## Running tests

### By marker

```console
pytest -m unit                       # Fast feedback during development
pytest -m smoke                      # Full-solver smoke
pytest -m "unit and not slow"        # CI push tier
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
| Push / PR (`ci_tests.yml`) | `unit and not slow` | < 5 min | Yes (unit tier); uploaded to Codecov under flag `ci` from `ubuntu-latest` + py3.12 only |
| Nightly cron + push to main (`nightly.yml`, 02:30 UTC) | `unit or smoke or slow` | < 90 min | Yes (full suite); uploaded to Codecov under flag `nightly` |
| Manual `workflow_dispatch` | as above | < 90 min | Yes |

Push CI runs the unit tier only because each smoke test executes a full `EntropySolver` call (5 to 15 min on a 2-vCPU runner under coverage instrumentation). The nightly tier carries the canonical 95% coverage floor; the per-push upload is a fast-feedback companion view of the unit subset.

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

Open `htmlcov/index.html` to inspect line-by-line coverage. Both push CI
(unit tier) and the nightly (full suite) emit `--cov-report=xml` and upload
to Codecov under separate flags (`ci` and `nightly`). The project floor is
95%, enforced via `[tool.coverage.report].fail_under` in `pyproject.toml`.

## Linting

Before committing, format and check all files:

```console
ruff check --fix src/ tests/ tools/
ruff format src/ tests/ tools/
```

The local ruff (often 0.12.x) and the CI ruff (0.15.x) sometimes disagree on
formatting drift; CI is canonical. Run BOTH `ruff check` and `ruff format`
before pushing — `format` does NOT catch lint rules like `E402` misplaced
imports.
