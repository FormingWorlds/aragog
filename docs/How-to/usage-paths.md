# Standalone vs PROTEUS-integrated usage

Aragog can be driven in two clearly distinct ways. Choose the one that matches your task before reading the rest of the How-to and Tutorial pages.

!!! tip "Pick a path"
    | If you want to | Use the | Entry point |
    |----------------|---------|-------------|
    | Run a single mantle problem with a hand-written TOML | **standalone** path | `aragog.solver.EntropySolver` |
    | Run a coupled atmosphere-interior simulation | **PROTEUS-integrated** path | `proteus.interior.aragog` wrapper |

## Standalone path

The standalone path drives Aragog directly via the Python API. The user supplies a TOML configuration file and a directory of pressure-entropy EOS tables; the solver returns a `SolverOutput` dataclass.

```python
from aragog.solver import EntropySolver

solver = EntropySolver.from_file(
    filename="my_config.toml",
    eos_dir="data/lookup",
)
solver.initialize()
solver.set_initial_entropy(2900.0)
solver.solve()
out = solver.get_state()
```

This is the right path when you want to:

- Test or develop the solver itself.
- Reproduce a published Aragog or SPIDER calculation against a fixed boundary condition.
- Write your own outer driver that loops over masses, compositions, or initial entropies.

The full standalone walkthrough is in [Tutorials: First run](../Tutorials/firstrun.md). The TOML schema is in [How-to: Configuration](configuration.md).

The standalone CLI exposes a single `aragog` command-line entry point; it has no production subcommands. Run the Python API directly from scripts or notebooks.

## PROTEUS-integrated path

When Aragog runs as part of the PROTEUS framework, the configuration is built programmatically by the wrapper at `src/proteus/interior/aragog.py` in the PROTEUS repo. The wrapper:

- Maps PROTEUS config keys (`config.interior_energetics.aragog.*`, `config.interior_struct.*`) onto Aragog's `Parameters`.
- Selects the EOS table directory: `output/data/aragog_pt/` for PALEOS-generated tables, or a SPIDER-bundled directory for parity runs.
- Supplies a four-column external mesh file (`output/data/{spider,zalmoxis}_mesh.dat`) when `eos_method = 2`.
- Drives `EntropySolver.set_initial_entropy()` from the previous step's profile.
- Wraps `solve()` in a retry ladder that calls `set_initial_dSdr_cmb()` and a tolerance-relaxation knob to recover from `status = -1` failures.
- Reads `SolverOutput` back into the PROTEUS `Interior_t` state and writes the per-iteration `int.nc` snapshot.

This is the right path for production magma-ocean evolution, atmosphere-interior coupling, and any paper-quality CHILI run. The TOML sections from the [standalone Configuration](configuration.md) page are **not** read in this mode; PROTEUS reads its own `[interior_energetics.aragog]` block.

For the PROTEUS recipe, see the PROTEUS documentation at [proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS) (in particular the coupled-CHILI tutorial under Tutorials).

!!! warning "Do not mix the two paths in one run"
    Setting both an Aragog-side TOML and a PROTEUS-side `[interior_energetics.aragog]` block in the same simulation is not supported. PROTEUS bypasses the file-based path entirely; the standalone TOML keys are silently ignored.

## What is shared between the two paths

The numerical core is identical. Both paths:

- Use `EntropySolver` as the integrator wrapper.
- Read the same SPIDER-format pressure-entropy EOS tables.
- Return a `SolverOutput` dataclass with the same fields.
- Apply the same boundary-condition logic and the same heat-transport switches.

The split is purely at the configuration boundary: in the standalone path, the user writes the TOML; in the PROTEUS-integrated path, the wrapper builds the equivalent dict programmatically.

## Cross-references

- [Tutorials: First run](../Tutorials/firstrun.md): standalone walkthrough.
- [How-to: Configuration](configuration.md): TOML schema (standalone) and key semantics (shared).
- [How-to: Usage](usage.md): standalone Python API reference.
- [PROTEUS documentation](https://proteus-framework.org/PROTEUS): coupled-run setup, CHILI configs, PROTEUS-side schema.
