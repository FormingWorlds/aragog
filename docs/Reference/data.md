# Reference data

Aragog's entropy solver requires a directory of pressure-entropy (P-S) equation-of-state tables. This page documents the file format, the canonical PALEOS data source, and the optional external mesh file used when the structure profile is supplied from outside (typically by the Zalmoxis structure solver).

## Pressure-entropy EOS tables

The directory passed to `EntropySolver.from_file(eos_dir=...)` (or referenced through PROTEUS) must contain ten files in the SPIDER-format $(P, S)$ layout; the `thermal_exp` pair in the table is optional:

| File | Format | Description |
|------|--------|-------------|
| `temperature_solid.dat` | 2D P-S grid | $T(P, S)$ for the solid phase |
| `temperature_melt.dat` | 2D P-S grid | $T(P, S)$ for the liquid phase |
| `density_solid.dat` | 2D P-S grid | $\rho(P, S)$ for the solid phase |
| `density_melt.dat` | 2D P-S grid | $\rho(P, S)$ for the liquid phase |
| `heat_capacity_solid.dat` | 2D P-S grid | $c_p(P, S)$ for the solid phase |
| `heat_capacity_melt.dat` | 2D P-S grid | $c_p(P, S)$ for the liquid phase |
| `adiabat_temp_grad_solid.dat` | 2D P-S grid | $(\partial T/\partial P)_S$ for the solid |
| `adiabat_temp_grad_melt.dat` | 2D P-S grid | $(\partial T/\partial P)_S$ for the liquid |
| `thermal_exp_solid.dat`, `thermal_exp_melt.dat` (optional) | 2D P-S grid | $\alpha(P, S)$ of each phase; loaded by `EntropyEOS` when present but not read by the solver, which derives $\alpha$ from the identity $\rho c_p (\partial T/\partial P)_S / T$ ([thermal expansivity of a single phase](../Explanations/phase_transitions.md#thermal-expansivity-of-a-single-phase)) |
| `solidus_P-S.dat` | 2-column $(P, S)$ | Solidus entropy at each pressure |
| `liquidus_P-S.dat` | 2-column $(P, S)$ | Liquidus entropy at each pressure |

### Quick start: bundled test data

A small SPIDER-format set is published as a release asset and used by the nightly CI. It is the fastest way to follow the [first-run tutorial](../Tutorials/firstrun.md) without a full PROTEUS / PALEOS install:

```bash
mkdir -p /tmp/aragog-test-data
curl -sL -o /tmp/spider_eos.tar.gz \
    https://github.com/FormingWorlds/aragog/releases/download/test-data-v1/spider_eos_test_data.tar.gz
tar xzf /tmp/spider_eos.tar.gz -C /tmp/aragog-test-data/
```

The unpacked directory `/tmp/aragog-test-data/spider_eos/` is a drop-in `eos_dir`. Set `ARAGOG_TEST_EOS_DIR=/tmp/aragog-test-data/spider_eos` so the gated unit + smoke tests pick it up automatically.

### File layout

The 2D tables follow SPIDER's text format: a header followed by a flat list of three-column $(P, S, Q)$ rows on a regular pressure-entropy grid.
The first header line carries the row count and grid dimensions in the form `# n_header n_P n_S`.
The last header line carries three SI scaling factors `# P_scale S_scale Q_scale` that the loader multiplies into each non-dimensional column to recover SI units (pressure in Pa, entropy in J/kg/K, value in the property's SI unit).
Inside the data block the entropy axis varies slowest and the pressure axis varies fastest, so the flat list contains $n_S$ blocks of $n_P$ rows each.

The `solidus_P-S.dat` and `liquidus_P-S.dat` files are two-column $(P, S)$ in the same convention (header with scaling factors, then non-dimensional values), with $P$ monotonically increasing.

The grid must be rectangular: phase-filtered tables (where each phase's grid is restricted to its own valid region) cause `scipy.interpolate.RegularGridInterpolator` to fall back to slow unstructured interpolation and should not be used.

### Validity masks

In `eos_dir`, next to the tables, `valid_mask_melt.dat` and `valid_mask_solid.dat` mark each node of the phase's temperature table: 1 where the generator had a valid EOS value, 0 where it filled the node or a property had no finite value. Each file is a 2-D array of 0 and 1 with one row per entropy node and one column per pressure node, shape $(n_S, n_P)$, with `#` comment lines allowed. Every column needs at least 1 valid node, and the $P$ and $S$ axes of the temperature table must rise strictly. A mask that breaks a rule of this section (shape $(n_S, n_P)$, values 0 and 1 only, a valid node in every column, strictly rising axes) raises `ValueError` when the first call reads it. The masks are optional; the [EOS domain check](../How-to/solver-method-tuning.md#lookups-outside-the-valid-window-of-the-tables) reads them.

### Canonical source: PALEOS

In the PROTEUS coupled path the tables are produced by the PALEOS multiphase EOS framework (Attia et al. (2026)[^cite-attia2026]) from a configured P-T melting curve and the Wolf & Bower (2018)[^cite-wolfbower2018] RTpress liquid EOS. The PROTEUS wrapper writes the resulting tables into `output/data/aragog_pt/` at construction time and points `eos_dir` there.

### Table cache

Aragog caches parsed ASCII tables into uncompressed NumPy binary (`.npz`) archives to avoid repeated text parsing during evaluator initialisations and runs.
Each cache archive stores the parsed array, format version, skipped row count, and a BLAKE2b digest of the source file.
If the source file changes or the skipped row count differs, the cache is automatically invalidated and regenerated.
The cache directory must be writable only by trusted users.

Cache files are resolved using the following location rules:

1. **User override**: When the environment variable `ARAGOG_TABLE_CACHE_DIR` is set, cache files are placed in that directory with a unique digest name `<digest>.npz`.
2. **FWL data root**: When the source table is located under `$FWL_DATA`, cache files are written to `$FWL_DATA/cache/tables/<digest>.npz` to preserve managed dataset trees.
3. **Beside source**: For any other table (such as run-directory tables generated by PROTEUS), the cache is written in the same directory as `<filename>.cache.npz`.

If a cache file cannot be written due to filesystem permissions or disk limitations, Aragog logs a message, skips later cache writes in that directory, still reads valid cache files there, and returns the parsed array directly.

## External mesh file (`eos_method = 2`)

When the mesh `eos_method` is set to 2, Aragog reads a four-column external file and uses it for the pressure, density, and gravity profiles instead of the analytic Adams-Williamson EOS:

| Column | Unit | Description |
|--------|------|-------------|
| `r` | m | Radial coordinate |
| `P` | Pa | Pressure at that radius |
| `rho` | kg/m³ | Density at that radius |
| `g` | m/s² | Gravitational acceleration at that radius |

The radius column must be monotonically increasing and span at least 75% of the mantle thickness $D = R_\mathrm{outer} - R_\mathrm{inner}$.
This is a *coverage tolerance* on the external table, not a statement that the structure is only defined over part of the domain.
Aragog's own mesh always covers the full mantle shell; the 75% guard allows the external EOS table to undershoot the mantle bounds by up to 25% without raising (typical use: a resumed run where the Zalmoxis re-solve disagrees with the saved mesh at the single-ULP level near the outer/inner radii).
Across that small gap `np.interp` extrapolates linearly from the nearest in-range value.
A table that covers less than 75% is rejected with `External EOS radius range ... inconsistent with mesh bounds ...` because at that point the disagreement is no longer a numerical-precision artefact.

The PROTEUS wrapper writes this file from the Zalmoxis structure solver into `output/data/zalmoxis_output.dat` (Zalmoxis path) or `output/data/spider_mesh.dat` (SPIDER-static path).
When `eos_method = 2` the per-node gravity column overrides the scalar `gravitational_acceleration` config key, so the mixing-length theory and the gravitational-separation flux pick up the radial dependence of $g$ automatically.

## Adams-Williamson EOS (`eos_method = 1`)

The default pressure profile is analytic Adams-Williamson:

$$
\rho^*(P) = \rho^*_\mathrm{top}\exp(P/K_S),\qquad \frac{dP}{dr} = -\rho^*\,g,
$$

closed under constant gravitational acceleration. Parameters are set in the `[mesh]` section:

| Key | Symbol | Description |
|-----|--------|-------------|
| `surface_density` | $\rho^*_\mathrm{top}$ | Mantle density at the surface |
| `adiabatic_bulk_modulus` | $K_S$ | Bulk modulus |
| `adams_williamson_beta` | $\beta$ | A-W exponent; `0.0` derives it from $K_S$ |
| `gravitational_acceleration` | $g$ | Constant gravity |
| `surface_pressure` | $P_\mathrm{top}$ | Atmospheric overburden added to the integration |

The Adams-Williamson path requires no external data and is the default for standalone testing.

## Core tables (`profile_source = "paleos"`)

With `profile_source = "paleos"` the core module reads 2 files, named under `[boundary_conditions.core_module_params]` ([a core on the PALEOS isentrope](../Explanations/core_tabulated.md)).

**Liquid iron table, `core_liquid_table`.** The liquid side of the PALEOS iron EOS that Zalmoxis writes, `2phase_iron_liquid_<sha>_<build>_<grid>.dat`. Each row holds `P` (Pa), `T` (K), `rho` (kg m$^{-3}$), `u` (J kg$^{-1}$), `s`, `cp` and `cv` (J kg$^{-1}$ K$^{-1}$), `alpha` (K$^{-1}$), `nabla_ad` (dimensionless) and a phase word, on a full rectangle in pressure and temperature. The `# Kind:` line of the header must name `liquid`. An `.npz` cut of the table holds the axes `P_Pa` and `T_K`, the 7 properties as 2-D arrays with one row per pressure, and `header`. Pressure rows with cells that did not converge (NaN) at the top or bottom of the table are dropped when the table is read; a NaN cell inside the table stops the read. Aragog interpolates the table with a bicubic spline in $\log_{10} P$ and $\log_{10} T$.

**Core arrays, `core_profile_file`.** An `.npz` with `r` (m), `m` (kg), `P` (Pa), `rho` (kg m$^{-3}$) and `g` (m s$^{-2}$) from the centre to the CMB, as Zalmoxis returns them in `result['core']`. The other Zalmoxis keys, such as `T`, `phase` and `M_ic`, are accepted and not used. The density must integrate to the node mass within $10^{-2}$.

The test data in `tests/validation/data/` hold files of both kinds with their provenance; `tools/verification/make_core_liquid_cut.py` cuts a Zalmoxis table to a pressure and temperature box.

## Configuration keys at a glance

The relevant configuration keys are summarised here for cross-reference; the full schema is in [How-to: configuration](../How-to/configuration.md).

| Key | Section | Sets |
|-----|---------|------|
| `eos_dir` (constructor argument) | -- | Directory containing the P-S tables |
| `eos_method` | `[mesh]` | `1` = Adams-Williamson; `2` = external file |
| `eos_file` | `[mesh]` | Path to the four-column external mesh file |
| `surface_density`, `adiabatic_bulk_modulus`, `adams_williamson_beta`, `gravitational_acceleration`, `surface_pressure` | `[mesh]` | Adams-Williamson parameters |
| `profile_source` | `[boundary_conditions.core_module_params]` | `"gaussian"` (default) or `"paleos"` |
| `core_liquid_table` | `[boundary_conditions.core_module_params]` | Path of the liquid iron table (`.dat` or `.npz`) |
| `core_profile_file` | `[boundary_conditions.core_module_params]` | Path of the core arrays (`.npz`) |

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre, *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*, Astronomy & Astrophysics, 713, A225, 2026. [arXiv:2605.03741](https://arxiv.org/abs/2605.03741).
[^cite-wolfbower2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59–74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
