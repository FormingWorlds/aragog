# Aragog test instructions

<!-- fwl-tests-core:begin sha256=0d210e346c572a96 -->
## Test rules shared by the PROTEUS ecosystem

Each test file starts with a module-level tier marker and a timeout. CI selects tests by marker, so a file without one runs in no CI job; the timeout stops a hang, it is not a target.

```python
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
```

| tier | what it tests | timeout |
|---|---|---|
| `unit` | Python logic with the heavy physics mocked; aim for < 100 ms | 30 s |
| `smoke` | the real binaries or solvers, one step, low resolution | 60 s |
| `integration` | several modules coupled | 300 s |
| `slow` | full physics validation | 3600 s |

Each test carries exactly one tier marker, so that the tier filters select it in the one CI job meant for it; a function marker for a second tier breaks that. Split a file whose tests need different tiers. `skip` excludes a test from every CI job.

Every new test covers an edge case (a boundary value, an empty input, an extreme physical parameter), exercises the error contract (a documented exception with the check that no side effect ran, a guard, a clamp, or the limit input of the formula), and asserts values that do not follow trivially from the implementation. A pinned value of 1 that every exponent reproduces checks nothing.

`python tools/check_test_quality.py --check` fails when the count of any rule rises above `tools/test_quality_baseline.json`; the offenders it prints are the first few of that rule in the tree, not necessarily yours. The rules: a file without a tier marker, a test without a docstring, a test with one assertion or none, a weak assertion as the only one (`is None`, `is not None`, `> 0`, `len(...) > 0`, `isinstance`), `==` next to a non-zero float literal, and an optional dependency imported without `pytest.importorskip`. `bash tools/validate_test_structure.sh` runs the repository's own structure check; the repository part of this file says what it checks.

Physics tests carry markers so their coverage is tracked apart from line coverage:
- `@pytest.mark.physics_invariant` on each test function that asserts a conservation law, a bound (T > 0, fractions in [0, 1]), a monotonicity or symmetry, or a pinned value with a discrimination guard. The marker goes on the function, not the module: structural tests in the same file do not carry it.
- `@pytest.mark.reference_pinned` (together with `physics_invariant`) on a test that pins a published benchmark, an analytical limit or a cross-implementation result; cite the paper, table or figure in the docstring.
- A discrimination guard asserts that the most plausible wrong formula (a missing factor, a swapped exponent, the wrong unit) gives a result outside the tolerance of the pinned value.

Floats: compare with `pytest.approx` or `np.testing.assert_allclose` and a tolerance you can justify from the method, not the one that makes the test pass.

Mocks: mock at the narrowest scope (the one external call), and return physically plausible values, so the code under test runs its real branches. Set random seeds and write files only under `tmp_path`.

A module-level constant read from an environment variable at import time does not change with `monkeypatch.setenv`; patch the constant with `monkeypatch.setattr`.
<!-- fwl-tests-core:end -->

## Aragog specifics

Aragog has no `tools/check_test_quality.py`; `bash tools/validate_test_structure.sh` checks that every test (by function, class or module-level `pytestmark`) carries at least one of `unit`, `smoke`, `integration`, `slow` or `skip`, and CI runs it before the tests; it does not catch a second tier marker, and some tests carry two (`pytest --collect-only -m "(unit and (smoke or integration or slow)) or (smoke and (integration or slow)) or (integration and slow)"` lists them). Markers registered in `pyproject.toml`: `unit`, `smoke`, `integration`, `slow`, `physics_invariant`, `reference_pinned`; `addopts = "--strict-markers"` applies to a local run, while the CI commands pass `-o "addopts="`, so CI warns on an unregistered marker instead of failing. Most test files set no timeout and `pyproject.toml` sets no default, so the timeout part of the shared rule is not in place here yet; the nightly passes `--timeout`.

- PR CI runs `pytest -m "unit and not slow" -n auto -o "addopts="`. The nightly runs `-m "unit or smoke or integration or slow"` with coverage, `tests/test_core_verification_page.py` in its own job in one process (`-p no:xdist`, its stable-layer case runs for many minutes), and combines the coverage against `fail_under = 90` in `pyproject.toml`.
- The SPIDER P-S tables come from `ARAGOG_TEST_EOS_DIR`, else `$FWL_DATA/aragog/spider_eos`, else a parity cache next to the repository (`tests/conftest.py`). Without them every test marked `needs_eos` skips: read the skip count of a run, a green run without the tables tests far less. `pytest_collection_finish` parses the tables once before the first test, outside the per-test timeouts.
- `tests/conftest.py` holds the table helpers (`entropy_eos_copy`, `entropy_eos_jax`, `make_mesh`), the `helper` fixture (the bundled configs and the reference data in `tests/reference/`) and the autouse `_isolated_package_logger` fixture; other fixtures are in the test files. The autouse fixture restores the `fwl.aragog` logger after each test and closes the handlers the test added: `aragog run` through `CliRunner` sets it to DEBUG with a file handler, and under DEBUG the mesh formats a whole array on every state update.
- `tests/test_core_verification_page.py` checks each tagged number of `docs/Validation/core.md` against `docs/figures/vv/core_verification_values.json` and reruns the items of `tools/verification/run_core_verification.py` it names; a change to the page, the values file or the script goes with that test.
- The smoke tests of the JAX solver need `diffrax`, which the nightly installs and no extra lists.
- The shared rule pairs `reference_pinned` with `physics_invariant`; 8 of the 27 `reference_pinned` tests here carry `reference_pinned` alone (the core page tests, a JAX parity test and three single-value pins).
- Tests that take the `validation_plot` fixture need `proteus-mpl` (in the `test` extra and in the CI installs); `pytest -m reference_pinned --validation-plots` writes their plot pairs to `docs/assets/validation/`.
