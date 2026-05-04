# Code architecture

This page describes the package structure under `src/aragog/` and the data flow between modules during a single time integration.

## Package layout

```
src/aragog/
├── __init__.py            # Package version, logging helpers, exposes EntropySolver
├── cli.py                 # Click CLI entry point (minimal: bare group)
├── parser.py              # Parameters dataclass and TOML/INI parsing
├── utilities.py           # Small utility functions, type aliases
├── cfg/                   # Bundled example configurations (.cfg, .toml)
│
├── config/                # attrs-based configuration sub-classes (Config facade)
│   ├── __init__.py        # Config.from_toml, .from_dict, .from_file → Parameters
│   ├── boundary.py        # BoundaryConfig (BC modes, core_bc selector, UTBL)
│   ├── energy.py          # EnergyConfig (transport switches, solver_method, JAX)
│   ├── initial_condition.py
│   ├── mesh.py            # MeshConfig (Adams-Williamson, mass coordinates, eos_file)
│   ├── phases.py          # PhaseConfig (solid/liquid), MixedPhaseConfig (cp_blend, const_*)
│   ├── radionuclides.py
│   ├── scalings.py        # Inert; values overridden to 1.0 by parser
│   └── solver.py          # SolverConfig (atol, rtol, time window)
│
├── eos/                   # Pressure-entropy equation of state
│   ├── __init__.py        # exports EntropyEOS, EntropyPhaseEvaluator
│   ├── entropy.py         # EntropyEOS: P-S table loader and bilinear lookup
│   └── entropy_phase.py   # EntropyPhaseEvaluator: SPIDER-parity phase blending
│
├── solver/                # Time integration and per-RHS state assembly
│   ├── __init__.py        # exports EntropySolver, EntropyState, BoundaryConditions, SolverOutput
│   ├── boundary.py        # Surface grey-body / UTBL, prescribed flux/T BCs
│   ├── entropy_solver.py  # EntropySolver: ODE driver, Radau/BDF/CVODE dispatch,
│   │                      #   nondimensionalisation, retry hooks, SolverOutput
│   ├── entropy_state.py   # EntropyState: per-RHS flux assembly, MLT, Jgrav, Jmix,
│   │                      #   phase-boundary caching
│   └── cvode_jax.py       # CVODE callbacks built from a JAX RHS+Jacobian factory
│
├── jax/                   # JAX-traceable replicas for the analytic-Jacobian path
│   ├── __init__.py
│   ├── eos.py             # EntropyEOS_JAX: equinox Module with jnp arrays
│   ├── phase.py           # PhaseParams, MeshArrays, compute_fluxes, compute_mlt
│   └── solver.py          # JAX-side dSdt and dSdt_energy_balance
│
├── mesh/                  # Staggered finite-volume mesh
│   ├── __init__.py        # Mesh class (basic + staggered nodes, dxidr, transforms)
│   ├── fixed_mesh.py      # FixedMesh (read-only radii, areas, volumes, mixing length)
│   └── pressure_eos.py    # AdamsWilliamsonEOS, UserDefinedEOS (external eos_file)
│
└── output/                # Diagnostic helpers
    ├── __init__.py        # exports melt_fraction_global, rheological_front
    └── diagnostics.py     # Standalone diagnostic functions; no Output class
```

The legacy single/mixed/composite phase evaluator stack and the temperature-based `Output` class are not present in the production path; results flow through the `SolverOutput` dataclass returned by `EntropySolver.get_state()`.

## Public API surface

The package's external API is intentionally narrow:

```python
from aragog import EntropySolver           # the only solver class
from aragog.solver import (
    EntropyState,                          # per-RHS state container
    BoundaryConditions,                    # surface grey-body, UTBL, prescribed
    SolverOutput,                          # dataclass returned by get_state()
)
from aragog.eos import EntropyEOS, EntropyPhaseEvaluator
from aragog.mesh import Mesh
from aragog.parser import Parameters       # config dataclass
from aragog.config import Config           # TOML/dict facade returning Parameters
from aragog.output.diagnostics import melt_fraction_global, rheological_front
from aragog import aragog_file_logger      # logger setup
```

Anything not in that list is internal: in particular `entropy_state`, `entropy_solver._dSdt_single`, the `aragog.jax` module, and the `cvode_jax` builder are implementation details that the PROTEUS wrapper and tests reach into when they need to, but external callers should treat as private.

## Key design patterns

### Configuration: facade over `Parameters`

The `Config` class in `config/__init__.py` is a facade. `Config.from_toml(path)`, `Config.from_dict(d)`, and `Config.from_file(path)` all return a `Parameters` dataclass (defined in `parser.py`); the attrs-based `BoundaryConfig`, `EnergyConfig`, `MeshConfig`, etc. provide schema validation but the solver operates on `Parameters` internally. PROTEUS uses the dict-driven path; standalone users typically use the file-driven path.

The `[scalings]` section is parsed for back-compatibility but every scaling is overridden to `1.0` on load: non-dimensionalisation now happens internally inside the solver around the integrator (see [Energy equation](energy_equation.md)) and is not user-configurable.

### EOS: single phase-aware evaluator

`EntropyEOS` loads the SPIDER-format pressure-entropy tables (one 2D grid per property per phase, plus two 1-D solidus/liquidus files). `EntropyPhaseEvaluator` wraps the EOS with the SPIDER `EOSEval_Composite_TwoPhase` blend, viscosity tanh transition, gravitational-separation velocity, and pressure-dependent latent heat. There is no abstract `PhaseEvaluatorProtocol`: the production path uses one concrete evaluator class.

### Solver: state, evaluator, integrator dispatch

- `EntropySolver` owns the ODE dispatch (Radau / BDF / CVODE), the nondimensionalisation layer, and the retry-ladder hooks (`set_initial_dSdr_cmb`, `_atol_sf`).
- `EntropyState` holds the phase, density, $T$, $c_p$, $\alpha$, $k$, and flux arrays; it is updated at each RHS evaluation by `EntropyState.update(entropy, time, ...)`.
- `EntropyPhaseEvaluator` is held by `EntropyState`; the mesh and BCs come from the `Evaluator` at construction time.

`EntropySolver.reset()` rebuilds the mesh and BCs from the current parameters but keeps the loaded EOS tables; this is what the PROTEUS retry ladder uses to back out and re-attempt a failed integration with relaxed tolerances.

### Mesh: precomputed transforms

The `Mesh` class precomputes:

- `d_dr_at_basic_nodes` matrix: the SPIDER-parity centred-difference operator in uniform $\xi$-space, then chain-ruled by $d\xi/dr$.
- `quantity_at_basic_nodes` and `quantity_at_staggered_nodes` matrices: linear interpolation between the two grids.
- `volume_average` operator: cell-volume-weighted reduction.
- The per-node `dxidr` array used by the gradient operator.

These are matrix-vector products inside the RHS, not loops.

### JAX path: opt-in analytic Jacobian

When `solver_method = "cvode"` and `use_jax_jacobian = true`, the PROTEUS wrapper installs a factory via `EntropySolver.set_jax_cvode_factory()`. The factory returns `(rhs_fn, jac_fn)` callbacks for CVODE built by tracing `aragog.jax.phase.compute_fluxes` with `jax.jacrev`. The numpy RHS path is otherwise untouched. The `aragog.jax` subpackage is only imported when this path is enabled; there is no passive JAX import at module load.

## Data flow during one RHS evaluation

```
CVODE / scipy solve_ivp
   │
   ▼
EntropySolver.dSdt(t_nd, y_nd)              ── nondimensional in/out
   │
   ▼  (rescale to physical units)
EntropySolver._dSdt_single(time, S_arr, *extra_state)
   │
   ▼
EntropyState.update(entropy, time, dSdr_cmb=…)
   │
   ├─ EntropyPhaseEvaluator → EntropyEOS  (P-S lookup; phase, T, ρ, cp, α, k)
   ├─ d_dr_at_basic_nodes(S)              (∂S/∂r at basic nodes)
   ├─ flux assembly: F_cond, F_conv, F_grav, F_mix
   └─ heating assembly: H_radio, H_tidal
   │
   ▼  (back in EntropySolver._dSdt_single)
flux divergence  →  dS/dt at staggered nodes
boundary BC  →  surface flux (grey-body / prescribed) and CMB closure
   │
   ▼  (rescale to nondim, append extra-state derivative if extended)
return dy_nd/dt
```

The retry ladder around `solve()` reads `get_current_dSdr_cmb()` after a failed attempt, increments `_atol_sf`, and calls `set_initial_dSdr_cmb()` before the retry. The PROTEUS wrapper drives this loop.

## Adding new functionality

- **New flux contribution.** Extend `EntropyState.update` to compute the flux, add a configuration key in `config/energy.py` and the corresponding field in `parser._EnergyParameters`, and gate the new contribution on the boolean flag inside `entropy_state.py`.
- **New core BC mode.** Add a branch in `EntropySolver.set_initial_entropy` (state-vector layout), `_dSdt_single` (the closure ODE for the extra state), `get_state` (final-time post-processing), and `_state_is_extended`. Update the validator in `config/boundary.py`.
- **New EOS source.** Either provide P-S tables in the SPIDER format and point `eos_dir` at them, or extend `EntropyEOS` to accept a different file layout. The EOS lookup is bilinear `RegularGridInterpolator`; non-rectangular grids (e.g. phase-filtered tables) cause scipy to fall back to slow unstructured interpolation and should be avoided.
- **New diagnostic.** Add a function to `output/diagnostics.py` and re-export it from `output/__init__.py`. If the diagnostic should appear in the standard output, add a field to `SolverOutput` and populate it in `EntropySolver.get_state`.
