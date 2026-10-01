# How to build tests

[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/aragog?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/aragog)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/aragog/badges/tests-total.json)](https://proteus-framework.org/testing)

This page is about *writing* a new test, by hand or with an LLM. For running
the existing suite see [Testing suite](../Explanations/testing.md).

## Decision tree: which marker?

The marker is the load-bearing decision. The PR gate (`ci_tests.yml`) runs
`-m "unit and not slow"` on every push and pull request; the nightly
(`nightly.yml`) runs `-m "unit or smoke or integration or slow"` on
schedule, push to main, and manual dispatch. An unmarked test is rejected
at the PR gate by `tools/validate_test_structure.sh` (see below). Pick the
strictest marker the test fits.

| Use ... | When ... |
|---|---|
| `@pytest.mark.unit` | < 100 ms, no real solver call. EOS lookup helpers, parser validators, phase-evaluator branches, JAX-vs-numpy parity on point inputs, regression pins on physical constants (permeability thresholds, Cardano cubic coefficients), isolated branch coverage with mocked density. |
| `@pytest.mark.smoke` | One full `EntropySolver.solve()` call at relaxed tolerance that finishes in seconds to minutes. Verifies the whole code path runs end to end. |
| `@pytest.mark.integration` | Real-physics integration against a published reference (PALEOS, SPIDER bit-parity) where the test's contract is "this implementation reproduces *that* result." Slower than smoke, faster than slow. **Zero tests carry this marker today**; the slot is reserved for tests added under it, and the nightly tier already includes `integration` in its filter so they will be picked up automatically. |
| `@pytest.mark.slow` | Multi-Myr cooling runs, tolerance-convergence studies, anything that takes tens of minutes per test. Manual only. |

Tests covering more than one tier carry a single marker matching the
dominant runtime. Do not double-mark. The four-marker scheme is the
internal CI granularity; the public-facing badge surface in the README
collapses smoke + integration + slow into a single "Integration Tests"
category for non-developer readers.

The marker-validation gate (`tools/validate_test_structure.sh`, run as a
step in `ci_tests.yml`) AST-parses every `def test_*` under `tests/` and
fails the PR if any test is missing a marker, propagated from function-,
class-, or module-level decorations. Local check before pushing:

```console
bash tools/validate_test_structure.sh
```

## Choosing a file

Naming convention: `test_<module>_<aspect>.py`, lower-snake, one module per
file where possible.

| Situation | Where the test goes |
|---|---|
| New unit test for an existing module | `test_<module>.py` if it exists, else create it. |
| Branch-coverage test for a hard-to-reach path | Add to `test_entropy_pytest.py` (existing module-by-module organisation) or create a focused file like `test_jax_radio_per_step.py`. |
| Failure-mode test (raises, validation errors) | Add to the module's existing test file; do not split failure tests into a separate suffix. |
| Conservation-law or first-principles regression | `test_entropy_verification.py`. |
| JAX/numpy parity test | `test_jax_<aspect>.py` (matches `test_jax_entropy.py`, `test_jax_utbl_cardano.py`, `test_jax_radio_per_step.py`). |
| Regression for a fixed bug | Add to the closest existing file; do not create one regression file per bug. |

## Float comparisons

Always `pytest.approx` or `np.testing.assert_allclose`, never `==`.

```python
assert result == pytest.approx(expected, rel=1e-6)
```

Choose the tolerance to match the physics, not the implementation. State the
chosen tolerance with a one-line comment naming the source or the limiting
factor.

## Fixtures

| Fixture | Defined where | Use when ... |
|---|---|---|
| `helper` | `tests/conftest.py` (module scope) | Provides `Helper` with shared `atol`/`rtol`/`get_cfg_file`/`get_reference_file` accessors for any test. |
| `shared_eos` | Module-local in EOS-dependent test files (e.g. `tests/test_entropy_solver_integration.py`, `tests/test_entropy_solver_eos_method2_smoke.py`) | The test needs an `EntropyEOS` instance built from the SPIDER-format pressure-entropy tables. Each file defines its own copy at module scope so the load amortises across that file's tests. |

EOS-dependent tests skip rather than fail when the table directory is
missing, via a per-file pattern:

```python
EOS_DIR = ...  # resolves ARAGOG_TEST_EOS_DIR / FWL_DATA / a parity-cache fallback
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)
```

Adding a new fixture: put it in `tests/conftest.py` if it is shared across
files; keep it module-local otherwise. Session-scope only when the fixture
is genuinely expensive to build (table loaders, solver calls). The
existing `shared_eos` copies are module-scoped because the EOS load is
fast enough at file granularity and the per-file isolation keeps the
collection-time skip pattern simple.

## Reference values

Every reference value asserted by a test must come from a primary source
cited inline. Examples:

```python
# SPIDER bc.c::core_BC v4 energy-balance formula at the basal cell, Earth IC.
# Bower (2018) §3 derivation; bit-parity to within float64 ULP.
assert F_cmb == pytest.approx(F_cmb_spider, rel=1e-12)

# Stefan-Boltzmann grey-body cooling: F = sigma * T^4 within 1% of analytic.
assert F_atm == pytest.approx(sigma * T_surf**4, rel=1e-2)
```

Do not "guess" a tolerance from a previous run. If you do not have a
reference, the test belongs in the unit tier with a synthetic input whose
expected output you can derive analytically.

## Discriminating test values

Choose input values that distinguish the correct formula from plausible wrong
formulas. A test of `F = sigma * T^4` at $T = 1$ K is useless because $T^1
= T^2 = T^4 = 1$. Test at $T = 300$ K and $T = 1500$ K, where the exponent
matters.

For any numerical function, ask: what are the 2 to 3 most plausible bugs
(off-by-one exponent, wrong sign, missing factor of 2, addition instead of
multiplication)? Ensure at least one test value distinguishes the correct
formula from these bugs.

## Mocking discipline

Mocking is appropriate when:

- The test isolates a non-EOS component (the ODE integrator, a parser
  validator, the energy-balance core BC) and EOS evaluation is a confounder.
- The point of the test is the analytic limit (constant density, zero source
  terms, frozen phase fractions).

Mocking is not appropriate when the test is meant to verify
EOS-table-dependent physics (mushy-zone width, phase routing, gravitational
separation). For those, exercise the real EOS path through `shared_eos`.

When mocking a physics function, ensure the mock returns physically plausible
values. A mock that returns 0.0 or 1.0 for everything can mask real bugs.

## Physical-invariant assertions

For integration and smoke tests that run actual simulations, verify:

- Total energy is conserved (within numerical tolerance).
- Mass fractions sum to 1.0 at all timesteps.
- Temperature is positive everywhere.
- Pressure monotonically increases with depth.
- The smooth-clipped melt fraction stays in $[0, 1]$.
- The gravitational-separation flux is divergence-free in mass.

These invariants hold for any valid simulation run, not just specific test
scenarios. They are the strongest class of test assertion because they catch
bugs regardless of the specific input.

## Comment hygiene

Inline comments and docstrings should explain *why* the test exists, never
*when it was added* or *what it used to do*. Acceptable:

```python
# Bower (2018) Eq. 3.4 derives the basal energy-balance cell flux from
# core enthalpy storage. A regression that drops the latent-heat term
# produces a -19% T_core offset against SPIDER.
```

Not acceptable:

```python
# Added in T2.3 to cover the rootfn bug we found on 2026-04-27.
# Previously this assertion was rel=1e-3; loosened to 3e-2 in commit abc1234
# after the BLAS-noise rerun.
```

History belongs in the commit message and the PR description, not in the
test source.

## A worked example

A unit test for the `combine_properties` helper in `aragog.utilities`,
which linearly blends two properties via a weight in `[0, 1]`:

```python
# tests/test_utilities.py
from __future__ import annotations

import pytest

from aragog.utilities import combine_properties


@pytest.mark.unit
def test_combine_properties_weight_zero_returns_property2():
    """weight = 0 must return property2 unchanged (no blend)."""
    assert combine_properties(0.0, 10.0, 20.0) == pytest.approx(20.0)


@pytest.mark.unit
def test_combine_properties_weight_one_returns_property1():
    """weight = 1 must return property1 unchanged (full blend)."""
    assert combine_properties(1.0, 10.0, 20.0) == pytest.approx(10.0)


@pytest.mark.unit
def test_combine_properties_blends_linearly_at_half():
    """weight = 0.5 must produce the arithmetic mean."""
    assert combine_properties(0.5, 10.0, 20.0) == pytest.approx(15.0)
```

Three unit tests, three branches (boundary 0, boundary 1, midpoint), all
fast, each with a one-line docstring stating the rationale.

## Anti-patterns

- **Forgetting the marker.** Tests without `@pytest.mark.unit` (or another
  marker) are invisible to CI. Run `pytest --collect-only -m unit | tail`
  after adding a test to confirm pickup.
- **Hardcoding paths.** Use `pathlib.Path(__file__).parent` or the project-
  provided helpers, never absolute paths.
- **Test ordering dependence.** Each test must pass in isolation. xdist
  reorders aggressively; relying on side-effects from a previous test is a
  bug, not a shortcut.
- **Asserting on log output.** Logs change for cosmetic reasons; assert on
  return values or state. Use `caplog` only when the log line is itself the
  contract (for example, a `phi_step_cap` rootfn fire log).
- **Sleeping or polling.** If a test needs a wait, the code under test has a
  race condition; fix that first.
- **Single-assert happy-path tests.** Add at least one edge case (boundary
  value, empty input, extreme physical parameter) and one physically
  unreasonable input that must raise or be handled.

## Suggested LLM prompt

When asking an LLM (Claude, Cursor, Copilot) to add or modify tests, paste
the prompt below at the start of the request along with the relevant source
file. The prompt encodes the PROTEUS and Aragog testing principles so the
generated test passes review without an iteration round.

````text
You are writing a pytest test for the Aragog entropy-form interior dynamics
solver (part of the PROTEUS ecosystem). Follow these rules strictly.

MARKERS (mandatory; the PR gate rejects unmarked tests via
tools/validate_test_structure.sh):
- @pytest.mark.unit: < 100 ms, no real solver call. Use for EOS lookup
  helpers, parser validators, phase-evaluator branches, JAX-vs-numpy parity,
  regression pins on physical constants, branch coverage.
- @pytest.mark.smoke: one full EntropySolver.solve() call at relaxed
  tolerance that finishes in seconds to minutes. Use for end-to-end smoke
  checks.
- @pytest.mark.integration: real-physics integration against a published
  reference (PALEOS, SPIDER bit-parity) where the test contract is "this
  implementation reproduces that result". Nightly tier.
- @pytest.mark.slow: multi-Myr cooling runs, tolerance-convergence studies,
  anything that takes tens of minutes per test. Manual only.
Pick the strictest marker the test fits. Do not double-mark.

FLOAT COMPARISONS:
- Always pytest.approx or np.testing.assert_allclose, never ==.
- Choose the tolerance to match the physics, not the implementation.
- State the source or limiting factor in a one-line comment.

REFERENCE VALUES:
- Cite the primary source inline. If no source is available, the test must
  use an analytically derivable synthetic input.
- Do not infer tolerances from a previous run.

DISCRIMINATING VALUES:
- Use inputs that distinguish the correct formula from plausible bugs (off-
  by-one exponent, sign error, missing factor of 2). Avoid T=1 K type
  degenerate inputs.

FIXTURES:
- Use `shared_eos` for the SPIDER-format pressure-entropy tables.
- Add new fixtures to tests/conftest.py only if shared across files.

MOCKING:
- Mock EOS only when the test isolates a non-EOS component or exercises an
  analytic limit. For EOS-table-dependent physics, use the real EOS path
  via `shared_eos`.

PHYSICAL INVARIANTS (smoke and slow tests):
- Energy and mass conservation within solver tolerance.
- Temperature positive, melt fraction in [0, 1].
- Pressure monotonically increasing with depth.

NAMING:
- Files: test_<module>.py, test_jax_<aspect>.py for JAX parity tests.
- Function names: snake_case, descriptive, no test_1 / test_2.

STYLE:
- `from __future__ import annotations` at the top of every file.
- One-line docstring stating the rationale for the test.
- Comments explain WHY, never WHEN added or what the code used to do.
- No project-tracking labels (T1.x, Stage X, ISO dates, commit SHAs).

ANTI-PATTERNS:
- No bare assertions on float equality.
- No hardcoded absolute paths.
- No reliance on test execution order.
- No assertions on log content unless the log line is itself the contract.
- No sleep / poll. If you need to wait, the code under test has a race.

OUTPUT:
- Produce only the test source. Do not modify the module under test.
- Place the new test in the appropriate existing file, or name a new file
  per the naming convention.
- After the test, list (a) which marker you chose and why, (b) the
  reference source for any literature value, (c) the tolerance and its
  justification.
````
