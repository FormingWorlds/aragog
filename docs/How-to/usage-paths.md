# Standalone usage

This page covers running Aragog directly via its standalone Python API. For driving Aragog from inside PROTEUS, see [Coupling Aragog to PROTEUS](proteus_coupling.md).

## When to use the standalone path

The standalone path is the right choice when you want to:

- Test or develop the solver itself.
- Reproduce a published Aragog or SPIDER calculation against a fixed boundary condition.
- Write your own outer driver that loops over masses, compositions, or initial entropies.
- Run isolated unit-style verifications without the full PROTEUS stack.

If you want a coupled atmosphere-interior simulation, do not use this path; PROTEUS bypasses Aragog's TOML loader entirely. Read [Coupling Aragog to PROTEUS](proteus_coupling.md) instead.

## Minimal example

The standalone path drives Aragog directly via the Python API. The user supplies a TOML configuration file and a directory of pressure-entropy EOS tables; the solver returns a `SolverOutput` dataclass.

```python
from aragog.solver import EntropySolver

solver = EntropySolver.from_file(
    filename='my_config.toml',
    eos_dir='data/lookup',
)
solver.initialize()
solver.set_initial_entropy(2900.0)
solver.solve()
out = solver.get_state()
```

The full standalone walkthrough is in [Tutorials: First run](../Tutorials/firstrun.md). The TOML schema is in [Configuration](configuration.md).

## CLI

The standalone CLI exposes a single `aragog` command-line entry point; it has no production subcommands. Run the Python API directly from scripts or notebooks.

## What is shared with the coupled path

The numerical core is identical. Both the standalone and PROTEUS-integrated paths:

- Use `EntropySolver` as the integrator wrapper.
- Read the same SPIDER-format pressure-entropy EOS tables.
- Return a `SolverOutput` dataclass with the same fields.
- Apply the same boundary-condition logic and the same heat-transport switches.

The split is purely at the configuration boundary: in the standalone path, the user writes the TOML; in the PROTEUS-integrated path, the wrapper builds the equivalent `Parameters` programmatically.

!!! warning "Do not mix the two paths in one run"
    Setting both an Aragog-side TOML and a PROTEUS-side `[interior_energetics.aragog]` block in the same simulation is not supported. PROTEUS bypasses the file-based path entirely; the standalone TOML keys are silently ignored when running under PROTEUS.

## Cross-references

- [Tutorials: First run](../Tutorials/firstrun.md): standalone walkthrough.
- [Configuration](configuration.md): TOML schema (standalone) and key semantics (shared).
- [Usage](usage.md): standalone Python API reference.
- [Coupling Aragog to PROTEUS](proteus_coupling.md): the PROTEUS-driven path.
