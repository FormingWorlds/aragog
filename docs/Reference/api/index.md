# API overview

This is the auto-generated API reference for Aragog's public interface. For the underlying physics, see the [model overview](../../Explanations/model.md); for the package layout in detail, see [code architecture](../../Explanations/code_architecture.md).

## Module overview

```
src/aragog/
├── config/                # Configuration: attrs sub-classes + Config facade
├── eos/                   # EntropyEOS, EntropyPhaseEvaluator
├── solver/                # EntropySolver, EntropyState, BoundaryConditions, SolverOutput
├── mesh/                  # Mesh, FixedMesh, AdamsWilliamsonEOS, UserDefinedEOS
├── jax/                   # Optional JAX-traceable replicas (analytic-Jacobian path)
├── output/                # Diagnostic helpers (rheological_front, total_enthalpy, volume_average)
├── parser.py              # Parameters dataclass and INI parser
├── cli.py                 # Click CLI (run, inspect, validate, show-config, new, list-configs, vnv)
└── utilities.py           # Utility functions and type aliases
```

The legacy single/mixed/composite phase evaluator stack and the temperature-based `Output` class have been replaced by the entropy-formulation modules; the public path goes through `EntropySolver` and `SolverOutput`.

## API reference

### Configuration

- [`aragog.config`](aragog.config.md) — `Config` facade with `.from_toml`, `.from_dict`, and `.from_file` constructors that all return a `Parameters` dataclass.

### Solver

- [`aragog.solver`](aragog.solver.md) — `EntropySolver` (the public solver class), `EntropyState`, `BoundaryConditions`, `SolverOutput`. Includes `EntropySolver.from_file`, `initialize`, `set_initial_entropy`, `set_initial_dSdr_cmb`, `set_initial_core_temperature`, `solve`, `get_state`, `reset`, `set_jax_cvode_factory`, and the retry-ladder hooks.

### EOS

- [`aragog.eos`](aragog.eos.md) — `EntropyEOS` (P-S table loader and bilinear interpolator) and `EntropyPhaseEvaluator` (SPIDER-parity phase blending, viscosity transition, gravitational-separation velocity, pressure-dependent latent heat).

### Mesh

- [`aragog.mesh`](aragog.mesh.md) — `Mesh` (staggered FV mesh with optional mass-coordinate transform), `FixedMesh`, `AdamsWilliamsonEOS`, `UserDefinedEOS` (external four-column mesh file).

### Output

- [`aragog.output`](aragog.output.md) — Standalone diagnostic helpers (`rheological_front`, `total_enthalpy`, `volume_average`). The primary output channel is `SolverOutput`, returned by `EntropySolver.get_state()`, which already exposes the global mantle melt fraction (`Phi_global` mass-weighted, `Phi_global_vol` volume-weighted); see the [solver page](aragog.solver.md) for its field-by-field description.

### JAX backend

- [`aragog.jax`](aragog.jax.md) — JAX-traceable replicas of the EOS, phase evaluator, and dSdt RHS, used to build analytic Jacobians via `jax.jacrev`. Only loaded when `solver.use_jax_jacobian = true`. See the [CVODE+JAX explanation](../../Explanations/cvode_jax.md) for when to opt in.

## What is not in the public API

The following modules are implementation details. They are reachable but should be treated as private:

- `aragog.solver.entropy_solver._dSdt_single`, `_solve_cvode`, `_build_jac_sparsity` (internal RHS and integrator dispatch).
- `aragog.solver.cvode_jax` (CVODE callback factory used by the JAX path).
- `aragog.parser._BoundaryConditionsParameters`, `_EnergyParameters`, etc. (the underscore-prefixed dataclasses are reached through `Parameters`, not directly).
