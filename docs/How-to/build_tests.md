# How to build tests

[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/FormingWorlds/aragog/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/aragog)

This page is about *writing* a new test, by hand or with an LLM. For running
the existing suite see [Testing suite](../Explanations/testing.md).

## Decision tree: which marker?

The marker is the load-bearing decision. CI runs `-m "unit and not slow"` on
every push and `-m "unit or smoke or slow"` nightly; an unmarked test is
invisible to CI. Pick the strictest marker the test fits.

| Use ... | When ... |
|---|---|
| `@pytest.mark.unit` | < 100 ms, no real solver call. EOS lookup helpers, parser validators, phase-evaluator branches, JAX-vs-numpy parity on point inputs, regression pins on physical constants (permeability thresholds, Cardano cubic coefficients), isolated branch coverage with mocked density. |
| `@pytest.mark.smoke` | One full `EntropySolver.solve()` call at relaxed tolerance that finishes in seconds to minutes. Verifies the whole code path runs end to end. |
| `@pytest.mark.slow` | Multi-Myr cooling runs, tolerance-convergence studies, anything that takes tens of minutes per test. Manual only. |

Tests covering more than one tier (for example, an integration test that also
exercises a unit-tier helper) carry a single marker matching the dominant
runtime. Do not double-mark.

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

| Fixture | Use when ... |
|---|---|
| `shared_eos` | The test needs the SPIDER-format pressure-entropy tables; opens once per session and hands the `EntropyEOS` instance to all dependents. Skips cleanly if `FWL_DATA` is unset. |

Adding a new fixture: put it in `tests/conftest.py` if it is shared across
files; keep it module-local otherwise. Session-scope only when the fixture
is genuinely expensive to build (table loaders, solver calls).

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

Adding a unit test for a hypothetical helper `clamp_phi(value)`:

```python
# tests/test_phase_clamp.py
from __future__ import annotations

import pytest

from aragog.eos.entropy_phase import clamp_phi


@pytest.mark.unit
def test_clamp_phi_negative_pins_to_zero():
    """Negative melt fraction is unphysical and must clamp to 0."""
    assert clamp_phi(-0.1) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_clamp_phi_above_one_pins_to_one():
    """Melt fraction greater than 1 is unphysical and must clamp to 1."""
    assert clamp_phi(1.5) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_clamp_phi_inside_range_is_identity():
    """Inside [0, 1] the value passes through unchanged."""
    assert clamp_phi(0.42) == pytest.approx(0.42, abs=1e-12)
```

Three unit tests, three branches, all fast, each with a one-line docstring
stating the rationale.

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

MARKERS (mandatory; CI-invisible without one):
- @pytest.mark.unit: < 100 ms, no real solver call. Use for EOS lookup
  helpers, parser validators, phase-evaluator branches, JAX-vs-numpy parity,
  regression pins on physical constants, branch coverage.
- @pytest.mark.smoke: one full EntropySolver.solve() call at relaxed
  tolerance that finishes in seconds to minutes. Use for end-to-end smoke
  checks.
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
