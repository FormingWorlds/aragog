# Aragog short instructions

The full instructions are in `AGENTS.md` (repository rules) and `tests/AGENTS.md` (test rules). This file repeats what a review or a chat needs when it cannot read them.

## Commands

```bash
pip install -e '.[test,jax]'
export ARAGOG_TEST_EOS_DIR=...   # SPIDER P-S tables, else the EOS tests skip
pytest -m "unit and not slow" -n auto -o "addopts="
ruff check src/ tests/ && ruff format --check src/ tests/
bash tools/validate_test_structure.sh
python tools/agents/check_agents_md.py
```

## Review points

- Every test carries exactly one tier marker (`unit`, `smoke`, `integration`, `slow`, or `skip`; the structure script catches only a missing one); a physics test asserts a conservation law, a bound or a published or analytical value, with a guard against the most plausible wrong formula.
- A change to anything PROTEUS reads (`EntropySolver`, its config fields, the output used by `proteus.interior_energetics.aragog.AragogRunner`) names the affected PROTEUS call sites in the pull request.
- Numbers on `docs/Validation/core.md` are checked by `tests/test_core_verification_page.py` against `docs/figures/vv/core_verification_values.json`; change them together.
- Comments: a block of 2 lines or fewer, never more than 4; commit messages and docs describe the current code, with no tool attribution and no em or en dashes.
