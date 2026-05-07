# Coupling to PROTEUS

This page is the **theory** of how Aragog plugs into a PROTEUS coupled run: where the energetics solver sits in the per-iteration sequence, what the wrapper does on init and on every time step, which `hf_row` keys it consumes and produces, and where the known landmines live. For the practical TOML recipe, see the [PROTEUS coupling how-to](../How-to/proteus_coupling.md).

When Aragog runs inside PROTEUS, the coupling is at the **function-call level**, not at the file level. PROTEUS does not write an Aragog TOML, does not invoke Aragog's standalone CLI, and does not read Aragog's NetCDF snapshots through the standalone path. The wrapper in `proteus/interior_energetics/aragog.py` builds an `EntropySolver` instance from the PROTEUS configuration, drives it forward by `interior_o.dt` per iteration, and reads the resulting `SolverOutput` back into the shared `hf_row` dictionary. State exchange across iterations happens through `hf_row`, the per-iteration NetCDF snapshot under `<outdir>/data/`, and (when Zalmoxis is the structure module) the external mesh file `zalmoxis_output.dat`.

## Where Aragog sits in the iteration

PROTEUS's main loop in `src/proteus/proteus.py` walks each module in a fixed order. Aragog is invoked twice per loop family: once on init (a one-shot `solve()` call to set up the entropy IC and warm any caches), and on every main-loop iteration as the interior thermal step. The relevant slice is:

```python
# Per main-loop iteration

# 1. Interior thermal evolution: SPIDER, Aragog, or dummy
run_interior(self.directories, self.config, self.hf_all,
             self.hf_row, self.interior_o, write_data=is_snapshot)

# 2. Advance simulation time using the achieved interior dt
self.hf_row['Time']     += self.interior_o.dt
self.hf_row['age_star'] += self.interior_o.dt

# 3. Optional structure refresh (Zalmoxis, gated)
update_structure_from_interior(...)

# 4. Orbit, stellar flux, escape, outgassing, atmosphere
...
```

Two ordering details matter for how the wrapper reads `hf_row`. First, the atmosphere step writes `F_atm` *before* Aragog runs, so Aragog's outer-boundary flux comes from the most recent atmosphere solve. Second, `Time` is advanced *after* Aragog returns its achieved `dt`, so any per-iteration time bookkeeping must distinguish between "PROTEUS time at solve entry" (`hf_row['Time']`) and "PROTEUS time at solve exit" (`hf_row['Time'] + dt`).

## What the PROTEUS wrapper does

The wrapper is split between two files. `src/proteus/interior_energetics/aragog.py` builds the `EntropySolver`, sets up the EOS-table directory, drives the per-iteration `solve()`, and writes the per-iteration NetCDF snapshot. `src/proteus/interior_energetics/wrapper.py` contains the orchestration shared with SPIDER (initial structure solve, equilibration loop, fall-back logic, mass-anchor check, mesh blending). The split is functional, not modular: the energetics-side file is the lower-level Aragog adapter, and the orchestration wrapper is the glue that ties Aragog to whichever structure module is active.

### 1. Initial setup

`AragogRunner.__init__` (`interior_energetics/aragog.py` line ~78) constructs the `EntropySolver`, sets the EOS directory (Zalmoxis-written `<outdir>/data/aragog_pt/` for PALEOS-generated tables, or a SPIDER-bundled directory for parity runs), and configures all of Aragog's `Parameters` from the PROTEUS schema. Outer-boundary mode 4 (prescribed flux) is always selected: Aragog consumes `hf_row['F_atm']` as the surface flux boundary condition, and the atmosphere module is responsible for self-consistently computing it from the surface temperature.

The initial entropy IC is set from `hf_row['ini_entropy']` on the first call, and from the previous iteration's S-field on subsequent calls (`read_last_Sfield(<outdir>, t)`). The CMB radius prefers `hf_row['R_core']` (set by Zalmoxis or SPIDER) when present and falls back to `core_frac * R_int` otherwise.

### 2. Per-iteration solve

`AragogRunner.run` (line ~1259) sets:

- `start_time = hf_row['Time']`
- `end_time   = hf_row['Time'] + dt`
- `outer_boundary_value = hf_row['F_atm']`
- `equilibrium_temperature = hf_row['T_eqm']`
- `outer_radius = hf_row['R_int']`
- `gravitational_acceleration = hf_row['gravity']`
- `surface_pressure = hf_row['P_surf']` (bar -> Pa internal conversion)

It then re-loads the previous S-field, calls `solver.solve()`, and packs the achieved `dt_actual = sol.t[-1] - sol.t[0]` plus surface fluxes back into the wrapper's `Interior_t` state. The achieved `dt` may be shorter than the requested window when the SUNDIALS root function fires for `phi_step_cap`; PROTEUS's outer loop then advances `Time` by the achieved value rather than the requested one.

### 3. Retry ladder

A non-converged `solver.solve()` (status $\ne 0$, T_core jump $> 1500$ K, or NaN propagation) routes through a retry ladder that progressively perturbs the entropy gradient at the CMB (`set_initial_dSdr_cmb`) and loosens tolerances within a bounded window. After eight consecutive failures (`_ARAGOG_MAX_CONSECUTIVE_FAILS`) the run aborts. On first success the streak resets to zero and is logged for post-hoc analysis.

### 4. NetCDF snapshot

When `write_data = True` (snapshot iteration), the wrapper calls `solver.output.write_state_nc(<outdir>/data/<iter>_int.nc)` to persist the full per-cell state for offline plotting and resume support. The snapshot contains the per-cell entropy, temperature, melt fraction, fluxes, and time-integrated energy diagnostics; see [Energy diagnostics](energy_diagnostics.md) for the schema.

## State exchange (`hf_row` reads and writes)

Aragog reads the current planet boundary state from `hf_row`, runs its time step, and writes back the time-evolved interior scalars.

### Reads from `hf_row`

| Key | Meaning | Used for |
|---|---|---|
| `Time` | Current PROTEUS time [yr] | `solver.parameters.solver.start_time`. |
| `F_atm` | Atmosphere outer-boundary flux [W m$^{-2}$] | `outer_boundary_value`, mode 4 (prescribed flux). |
| `T_eqm` | Equilibrium temperature [K] | Stefan-Boltzmann reference for outer-BC scaling. |
| `R_int` | Top-of-mantle radius [m] | `mesh.outer_radius`. |
| `R_core` | CMB radius [m] | `mesh.inner_radius`; falls back to `core_frac * R_int`. |
| `gravity` | Surface gravity [m s$^{-2}$] | `mesh.gravitational_acceleration` when `eos_method != 2`. |
| `P_surf` | Atmospheric pressure [bar] | `mesh.surface_pressure` (converted to Pa). |
| `ini_entropy` | Initial entropy IC [J kg$^{-1}$ K$^{-1}$] | `set_initial_entropy()` on first call only. |
| `core_density`, `core_heatcap` | Core thermal properties | Energy-balance core BC. |

### Writes to `hf_row`

| Key | Meaning |
|---|---|
| `T_magma` | Mantle potential temperature [K], mass-weighted from the entropy field |
| `T_surf` | Top-of-mantle temperature [K] |
| `T_core` | Core temperature [K], evolved via the energy-balance BC |
| `Phi_global` | Global melt fraction [0, 1], mass-weighted |
| `F_int` | Top-of-mantle energy flux [W m$^{-2}$] |
| `F_radio` | Radiogenic surface-equivalent flux [W m$^{-2}$] |
| `F_tidal` | Tidal surface-equivalent flux [W m$^{-2}$] (when `heat_tidal = true`) |
| `dt_actual` | Achieved time step [yr], may be less than requested when `phi_step_cap` fires |
| `step_dE_*_J` | Per-call energy budget integrals [J]: `F_int_J`, `F_cmb_J`, `Q_radio_J`, `Q_tidal_J` |

The full set of `hf_row` keys produced by Aragog is in `proteus.interior_energetics.aragog.AragogRunner.run` (lines ~1670 onwards). See also the [Energy diagnostics](energy_diagnostics.md) explainer for what the `step_dE_*_J` integrals mean and how they close the energy budget.

## Initial-condition contract

A coupled run's first iteration is the hardest. Aragog and the structure module must agree on $T(r)$, $P(r)$, $R_\mathrm{cmb}$, and $M_\mathrm{core}$ before the first time step. PROTEUS supports several IC modes (see the [Zalmoxis coupling theory page](https://proteus-framework.org/Zalmoxis/Explanations/proteus_coupling.html#initial-condition-anchors) for the full list); from Aragog's perspective, the contract is:

1. **Mesh consistency**: `mesh.inner_radius` and `mesh.outer_radius` must agree with the EOS-table $r$-range to relative tolerance $\max(1\,\mathrm{m},\ 10^{-9} \cdot \mathrm{span})$. Resumed runs sometimes hit single-ULP drift; if you see `External EOS radius range [..., ...] m inconsistent with mesh bounds [..., ...] m`, regenerate the mesh by re-running the structure module.
2. **Entropy IC**: the `set_initial_entropy()` value must lie inside the EOS table's S-range at the prevailing P. Out-of-range entropies are flagged at `solver._verify_entropy_ic`.
3. **F_atm sign**: PROTEUS sign convention is positive outward. A negative `F_atm` (atmosphere absorbing heat) is supported but rare; check the atmosphere module if it appears outside an active stellar-driven heating regime.

## External mesh handover

When `interior_struct.module = "zalmoxis"` and `interior_energetics.module = "aragog"`, Zalmoxis writes a five-column TSV (`r`, `P`, `rho`, `g`, `T`) to `<outdir>/zalmoxis_output.dat` on every successful re-solve. Aragog reads it inside `solver.reset()` to rebuild its mass-coordinate mesh.

The per-node gravity profile from the file overrides the scalar `gravity` value for `eos_method = 2`. This matters most for super-Earths, where the gravity profile is non-trivial across the mantle: the scalar surface-value override loses up to $\sim 30$% accuracy in the deep-mantle conductive flux estimate.

The `mesh_max_shift` cap (default 0.05 = 5%) in `interior_struct.zalmoxis` limits the per-update fractional radius shift to keep CVODE's BDF method tracking the new state without rebuilding the Newton matrix from scratch. The `blend_mesh_files` helper in `interior_energetics/spider.py` (used by both Aragog and SPIDER paths) post-modifies `zalmoxis_output.dat` to a partial blend when the requested shift exceeds the cap, and `dirs['mesh_shift_active']` is set so the [mesh-convergence trigger](https://proteus-framework.org/Zalmoxis/Explanations/proteus_coupling.html#dynamic-update-triggers) refires on the next iteration with the floor bypassed.

!!! note "Resume + ULP radius drift"
    `_validate_eos_radius_range` in `aragog/src/aragog/solver/entropy_solver.py` checks the EOS-table radius range against the live mesh bounds with relative tolerance $\max(1\,\mathrm{m},\ 10^{-9} \cdot \mathrm{span})$. When resuming from a saved snapshot, Zalmoxis recomputes mesh bounds from the live planet state while the EOS table radii are frozen at the launch-time mesh. A single-ULP mismatch is possible on a hot resume; regenerate the mesh by re-running the structure module rather than tightening this tolerance.

## EOS tables

Aragog's P-T tables for the production PALEOS coupling are written by Zalmoxis on the fly through `generate_aragog_pt_tables[_2phase]` (`zalmoxis.eos_export`, called from `proteus/interior_energetics/aragog.py`). Output goes to `<outdir>/data/aragog_pt/` as `{density, temperature, heat_capacity, adiabat_temp_grad, thermal_exp}_{melt, solid}.dat`, default resolution $200 \times 200$, P-range $[10^5\,\mathrm{Pa},\ \min(10^{13}\,\mathrm{Pa},\ 150 \cdot M_\oplus + 200\,\mathrm{GPa})]$.

!!! warning "Aragog tables must be strictly rectangular"
    Aragog's loader builds a `RegularGridInterpolator` over the (P, T) grid. Phase-filtering the input PALEOS table (for example, masking out points inside the mushy zone before writing) breaks the rectangularity assumption: scipy then silently falls back to unstructured (linear-ND) interpolation, which is roughly two orders of magnitude slower per call. The Zalmoxis-side generators write the full rectangle for both solid and melt files; do not inject a phase mask between them and disk.

The two-phase variant (`PALEOS-2phase:MgSiO3`) takes separate solid and liquid PALEOS tables and lets Aragog handle the latent-heat blend internally via the `mushy_zone_factor`. The unified-fallback path uses a single PALEOS table for both phases.

## Coupling pitfalls

The function-call coupling is robust at the file boundary, but several edges have known traps. New code paths should re-check these before claiming behavioural neutrality.

!!! warning "`prevent_warming` clamp is energy non-conserving"
    `config.planet.prevent_warming` (default `false`) gates an early ratchet `T_magma = min(new, prev)` in `interior_energetics/wrapper.py`. The clamp suits strictly-cooling regimes but **silently destroys the warming half of any heat-pump cycle** in a coupled magma-ocean run, producing an apparent "T_magma plateau" that is in fact an energy-leak bug. The clamp must remain at the `false` default for production runs. A separate runaway-T fallback (`interior_o.ic == 2` recovery path) remains active independently and is not affected. If you see `T_magma` byte-pinned across hundreds of consecutive iterations with any per-call energy residual growing without bound, check the `prevent_warming` flag first.

### `core_density` echo-back (SPIDER only)

When the structure module is Zalmoxis and the energetics module is **SPIDER** (not Aragog), Zalmoxis writes a self-consistent `hf_row['core_density']` from the converged structure, and SPIDER's call sequence then passes `-rho_core <value>` to the SPIDER C binary, which treats this density as authoritative. Aragog has no equivalent echo-back: its core thermal properties come from `hf_row['core_density']` and `hf_row['core_heatcap']` and feed directly into the energy-balance BC.

### Backend mismatch

`interior_energetics.aragog.backend = "numpy"` at production tolerances (`rtol = atol = 1e-9`) can trip Aragog's T_core-jump retry guard at the first dt jump. The CVODE finite-difference Jacobian carries enough noise that the BDF predictor occasionally overshoots and the retry ladder fires on what is otherwise a healthy step. The JAX path eliminates this. If you must use the numpy backend (for SPIDER-parity comparisons or on a JAX-incompatible machine), expect occasional retry events and budget wall time accordingly.

### `phi_step_cap` and atmosphere step size

The SUNDIALS root function for `phi_step_cap` returns at the exact zero-crossing, which can leave the integrator at a $dt_\mathrm{actual}$ much smaller than the requested window. PROTEUS's outer dt control reads `dt_actual` and shrinks the next-iteration request accordingly, but the atmosphere module (AGNI/JANUS) sees the same shrunk window and may need to re-equilibrate its temperature profile from a smaller flux differential. This is normal during the rheological transition; it is a signal to keep `phi_step_cap = 0.05`, not to disable it.

### Determinism

Coupled runs occasionally hit a class of numerical-fragility failures where the same config produces different results across launches: $\sim 10^{-7}$ floating-point noise in early helpfile rows compounds through Aragog's tight tolerances and lands the solver on a wrong P-S branch within $\sim 15$ iterations. PROTEUS pins BLAS thread counts at `cli.py` import time (`OMP_NUM_THREADS=1`, etc.); BLAS is not the only source of reduction-order non-determinism, JAX/XLA has its own threading model independent of OpenBLAS. The `--deterministic` flag intercepts itself in raw `sys.argv` *before* any heavy imports, sets `JAX_ENABLE_X64=1` and `XLA_FLAGS=--xla_cpu_enable_fast_math=false`, and self-re-execs once with a sentinel env var to prevent infinite re-exec.

Use sparingly: the flag has a small per-step cost (XLA fast-math disabled) and most coupled runs converge cleanly without it. It is intended for configs that show noise-floor divergence between launches, typically wet 1 $M_\oplus$ at IW+4 or reduced 1 $M_\oplus$ at IW-2.

## See also

- [PROTEUS coupling how-to](../How-to/proteus_coupling.md): TOML recipe and parameter reference.
- [Energy equation](energy_equation.md): the entropy-form RHS that Aragog integrates.
- [Energy diagnostics](energy_diagnostics.md): per-call energy-budget integrals (`step_dE_*_J`).
- [Aragog vs SPIDER](spider_comparison.md): formulation differences and parity guarantees.
- [Core BC modes](core_bc.md): the four core-mantle boundary conditions, with the `energy_balance` SPIDER bit-parity derivation.
- [PROTEUS framework documentation](https://proteus-framework.org/PROTEUS): top-level entry.

The wrapper itself lives in the PROTEUS repository, not in Aragog. The single source of truth for the symbol-level API is rendered from PROTEUS source via mkdocstrings:

- [`proteus.interior_energetics.aragog`](https://proteus-framework.org/PROTEUS/): `AragogRunner`, the per-iteration solve, NetCDF snapshot writers.
- [`proteus.interior_energetics.wrapper`](https://proteus-framework.org/PROTEUS/): `solve_structure`, `update_structure_from_interior`, the mass-anchor and fall-back logic shared with SPIDER.
- [`proteus.config._interior`](https://proteus-framework.org/PROTEUS/): the attrs schema for `[interior_energetics]` and `[interior_energetics.aragog]`. Every TOML key from the [coupling how-to](../How-to/proteus_coupling.md) maps to a field here.
