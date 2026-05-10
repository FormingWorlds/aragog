# Building a test for a new function

This tutorial walks through the full workflow for adding a unit test for a function you have just written.
It is aimed at new contributors and assumes you have already cloned and installed Aragog with the test extras.
For the full set of authoring rules (markers, fixtures, mocking, anti-patterns, LLM prompt template), see [How to build tests](../How-to/build_tests.md).

## What we will build

Imagine you are a new developer who has just added a small helper to `src/aragog/utilities.py` for computing the radiogenic-decay factor for a single isotope:

```python
# src/aragog/utilities.py
from __future__ import annotations

import math


def decay_factor(t: float, t0: float, half_life: float) -> float:
    """Fractional radioactive abundance at time t.

    Parameters
    ----------
    t : float
        Time of evaluation, in years.
    t0 : float
        Reference time at which the abundance is normalised to unity.
    half_life : float
        Half-life of the isotope, in years. Must be positive.

    Returns
    -------
    float
        ``exp(-ln 2 * (t - t0) / half_life)``.

    Raises
    ------
    ValueError
        If ``half_life`` is non-positive.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return math.exp(-math.log(2.0) * (t - t0) / half_life)
```

The function is small, has an analytic answer at three special points ($t = t_0 \Rightarrow 1$, $t = t_0 + \tau_{1/2} \Rightarrow 1/2$, $t = t_0 + 2\tau_{1/2} \Rightarrow 1/4$), and rejects an unphysical input (non-positive half-life).
That makes it an ideal candidate for a unit test.

## 1. Decide which marker

CI runs `pytest -m "unit and not slow"` on every push and `pytest -m "unit or smoke or slow"` nightly.
A test without a marker is invisible to CI.

`decay_factor` runs in microseconds and does not call the solver.
It belongs in the **unit** tier:

```python
@pytest.mark.unit
```

The full marker decision tree lives in [How to build tests](../How-to/build_tests.md#decision-tree-which-marker).

## 2. Decide where the test goes

Aragog tests live under `tests/` at the repo root and follow the convention `test_<module>.py`, one file per source module where possible.
For a helper in `src/aragog/utilities.py`, the natural home is `tests/test_utilities.py`.

If `tests/test_utilities.py` already exists, append the new tests to it.
If it does not, create the file with the standard PROTEUS-ecosystem header (`from __future__ import annotations`, an explicit `import pytest`, no project-level `__init__.py` required).

The full set of file-placement rules is in [How to build tests](../How-to/build_tests.md#choosing-a-file).

## 3. Write the tests

The minimum test set covers the analytic special points, a non-trivial intermediate point, and the failure mode.
Aim for one assertion per test, one-line docstring stating the *why*, and `pytest.approx` for every float comparison.

```python
# tests/test_utilities.py
from __future__ import annotations

import math

import pytest

from aragog.utilities import decay_factor


@pytest.mark.unit
def test_decay_factor_zero_offset_returns_one():
    """At t = t0 the abundance is normalised to 1 by construction."""
    assert decay_factor(t=0.0, t0=0.0, half_life=1.25e9) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.unit
def test_decay_factor_one_half_life_returns_one_half():
    """After one half-life the abundance is exactly 1/2."""
    tau = 1.25e9
    assert decay_factor(t=tau, t0=0.0, half_life=tau) == pytest.approx(0.5, rel=1e-12)


@pytest.mark.unit
def test_decay_factor_two_half_lives_returns_one_quarter():
    """After two half-lives the abundance is 1/4; distinguishes ``exp(-t/tau)``
    from the correct ``exp(-ln 2 * t / tau)``."""
    tau = 1.25e9
    assert decay_factor(t=2 * tau, t0=0.0, half_life=tau) == pytest.approx(0.25, rel=1e-12)


@pytest.mark.unit
def test_decay_factor_negative_offset_matches_analytic():
    """Pre-zero times grow back exponentially per the analytic formula."""
    expected = math.exp(math.log(2.0) * 0.5)  # t - t0 = -0.5 tau
    assert decay_factor(t=-0.5, t0=0.0, half_life=1.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_decay_factor_non_positive_half_life_raises():
    """A non-positive half-life is unphysical and must be rejected."""
    with pytest.raises(ValueError, match="half_life must be positive"):
        decay_factor(t=1.0, t0=0.0, half_life=0.0)
```

A few things to notice:

- **`@pytest.mark.unit` on every test.**
  Forgetting the marker is the most common reason a new test silently misses CI.
- **`pytest.approx` everywhere, never `==`.**
  Float equality is fragile; `rel=1e-12` is the right tolerance for an analytic identity at machine precision.
- **The two-half-lives test is *discriminating*.**
  Picking two distinct analytic anchors (one at $\tau$, another at $2\tau$) catches a wider class of bugs than a single anchor: a missing factor of 2, a sign flip, or a ratio bug that happens to be self-consistent at $t = \tau$ will diverge at $t = 2\tau$.
  Always pick test inputs that distinguish the correct formula from the most plausible wrong formulas.
- **The failure-mode test uses `pytest.raises` with `match=`.**
  Asserting on the error type alone passes when the function raises a different `ValueError` for a different reason; matching the message text fixes the contract.

## 4. Run the test locally

From the repo root:

```sh
pytest tests/test_utilities.py -v
```

To run only the new tests:

```sh
pytest tests/test_utilities.py -v -k decay_factor
```

To check that the marker is picked up by CI's collection rules:

```sh
pytest --collect-only -m "unit and not slow" tests/test_utilities.py | tail
```

If a test is missing from the collection output, the most likely causes are a missing `@pytest.mark.unit` decorator or a typo in the test function name (`pytest` only picks up names starting with `test_`).

## 5. Verify CI picks them up

After pushing the branch, the **Unit Tests** workflow runs `pytest -m "unit and not slow"` against ubuntu-latest and macos-latest with Python 3.11, 3.12, and 3.13.
The test file should appear under one of the green checks; if it does not, the test was not discovered (see step 4).

The **Integration Tests** workflow runs nightly and adds `-m "smoke or slow"`; unit tests run there too as part of `unit or smoke or slow`.

## What's next

- Read [How to build tests](../How-to/build_tests.md) for the full set of authoring rules: file placement, fixtures, mocking discipline, reference values, comment hygiene, and the LLM prompt template for test scaffolding.
- Read [Testing suite](../How-to/testing.md) for how the existing suite is organised, how to run only smoke or slow tiers, and how coverage is enforced at 95%.
- Browse `tests/` for examples close to your function: `tests/test_utilities.py` for helpers, `tests/test_jax_*.py` for JAX-vs-numpy parity, `tests/test_entropy_verification.py` for first-principles regressions.
