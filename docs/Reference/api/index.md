# API overview

This is a detailed overview of Aragog's API for the user's reference. If you want to understand the underlying model, please visit the [model overview](../../Explanations/model.md).

## Module overview

```
src/aragog/
├── config/               # Configuration (TOML + attrs validation)
├── solver/               # Solver, State, Evaluator, BoundaryConditions, InitialCondition
├── eos/                  # EOS evaluators (single, mixed, composite, properties)
├── mesh/                 # Staggered mesh, pressure EOS
├── output/               # Output, diagnostics, NetCDF writing, plotting
├── parser.py             # Legacy INI configuration parser
├── cli.py                # Command-line interface
├── core.py               # Core model helpers
├── phase.py              # Legacy phase evaluator
├── interfaces.py         # Legacy interface evaluator
└── utilities.py          # Utility functions
```

## API reference

### Configuration
- [`aragog.config`](aragog.config.md) - Config facade (from_toml, from_dict, from_file)

### Solver
- [`aragog.solver`](aragog.solver.md) - Solver class (initialize, solve, dTdt)
- [`aragog.solver.state`](aragog.solver.state.md) - State class (current temperature, fluxes, heating)

### EOS (Equation of State)
- [`aragog.eos`](aragog.eos.md) - EOS package (PhaseEvaluatorCollection, setup)
- [`aragog.eos.single_phase`](aragog.eos.single_phase.md) - Single-phase evaluator
- [`aragog.eos.mixed_phase`](aragog.eos.mixed_phase.md) - Mixed-phase evaluator
- [`aragog.eos.composite`](aragog.eos.composite.md) - Composite phase evaluator
- [`aragog.eos.properties`](aragog.eos.properties.md) - Property classes (constant, 1D lookup, 2D lookup)

### Mesh
- [`aragog.mesh`](aragog.mesh.md) - Staggered mesh, coordinate transforms

### Output
- [`aragog.output`](aragog.output.md) - Output class (NetCDF, plotting, diagnostics)
