# Code architecture

This page describes the refactored Aragog package structure in `src/aragog/`.

## Package layout

```
src/aragog/
├── __init__.py           # Package version, logging setup, public API (Solver, Output)
├── cli.py                # Click CLI entry point
├── parser.py             # Legacy Parameters dataclass (INI parser)
├── utilities.py          # Small utility functions, type aliases
├── core.py               # Core model helpers (initial/boundary condition support)
├── phase.py              # Legacy phase evaluator (backward compat)
├── interfaces.py         # Legacy interface evaluator (backward compat)
├── config/               # Modern TOML-based configuration
│   ├── __init__.py       # Config facade (from_toml, from_dict, from_file)
│   ├── boundary.py       # BoundaryConfig attrs class
│   ├── energy.py         # EnergyConfig attrs class
│   ├── initial_condition.py  # InitialConditionConfig attrs class
│   ├── mesh.py           # MeshConfig attrs class
│   ├── phases.py         # PhaseConfig, MixedPhaseConfig attrs classes
│   ├── radionuclides.py  # RadionuclideConfig attrs class
│   ├── scalings.py       # ScalingsConfig attrs class
│   └── solver.py         # SolverConfig attrs class
├── eos/                  # Equation of state and transport property evaluators
│   ├── __init__.py       # PhaseEvaluatorCollection, re-exports
│   ├── base.py           # Abstract base classes and protocols
│   ├── single_phase.py   # SinglePhaseEvaluator
│   ├── mixed_phase.py    # MixedPhaseEvaluator
│   ├── composite.py      # CompositePhaseEvaluator
│   ├── properties.py     # ConstantProperty, LookupProperty1D, LookupProperty2D
│   └── transport.py      # combine_properties(), tanh_weight()
├── solver/               # ODE solver and state management
│   ├── __init__.py       # Solver class (from_file, initialize, solve, dTdt)
│   ├── boundary.py       # BoundaryConditions (applies BCs to flux arrays)
│   ├── evaluator.py      # Evaluator (assembles mesh, phases, BCs, IC)
│   ├── initial.py        # InitialCondition (adiabatic or specified)
│   └── state.py          # State (current T, fluxes, heating, phase at all nodes)
├── mesh/                 # Spatial discretization
│   ├── __init__.py       # Mesh class (staggered mesh, transforms)
│   ├── fixed_mesh.py     # FixedMesh (node positions, areas, volumes)
│   └── pressure_eos.py   # EOS (AdamsWilliamsonEOS, UserDefinedEOS)
└── output/               # Post-processing and file output
    ├── __init__.py       # Output class (NetCDF writing, plotting)
    └── diagnostics.py    # Derived quantities (global melt fraction, rheological front)
```

## Key design patterns

### Configuration: facade over legacy parser

The `Config` class in `config/__init__.py` is a facade that constructs a legacy `Parameters` object from modern TOML input. The attrs-based sub-configs (e.g. `SolverConfig`, `MeshConfig`) provide validation, but the solver still operates on `Parameters` internally. This allows incremental migration without breaking the solver.

### EOS: protocol-based evaluators

The EOS package defines `PhaseEvaluatorProtocol` and `PropertyProtocol` as runtime-checkable protocols. Evaluators at different levels (single phase, mixed phase, composite) implement the same interface. This allows the solver and output code to work with any evaluator without knowing the concrete type.

The `CompositePhaseEvaluator` is the primary evaluator used in production. It delegates to single-phase or mixed-phase evaluators depending on the local temperature relative to the solidus and liquidus at each node.

### Solver: separate state from integration

The `Solver` class owns the ODE integration loop. The `State` class is updated at each right-hand-side evaluation with the current temperature array. The `Evaluator` class holds the mesh, phase evaluators, boundary conditions, and initial condition. This separation allows reuse: `Solver.reset()` reinitializes mesh and state while preserving loaded lookup tables.

### Mesh: transform matrices

The `Mesh` class precomputes two transform matrices:

1. **Derivative transform**: maps staggered-node quantities to basic-node gradients ($d/dr$)
2. **Quantity transform**: maps staggered-node quantities to basic-node interpolated values

These matrices are computed once at initialization and applied as matrix-vector products during each `dTdt` evaluation.

## Data flow during a timestep

1. `solve_ivp` calls `Solver.dTdt(time, temperature)`
2. `State.update(temperature, time)` computes phase properties and fluxes at all nodes
3. `BoundaryConditions.apply_flux_boundary_conditions(state)` modifies fluxes at boundaries
4. Flux divergence and heating rates are combined into $dT/dt$ per cell
5. The BDF integrator advances the temperature array

## Adding a new feature

**New property or EOS**: implement the `PropertyProtocol` interface (see `eos/properties.py`), then register it in the relevant evaluator.

**New heating source**: add the computation in `State`, expose it as a property, and include it in the `heating` sum. Add a toggle in `[energy]` config and `EnergyConfig`.

**New boundary condition type**: extend `BoundaryConditions` in `solver/boundary.py` with a new case in the dispatch logic.
