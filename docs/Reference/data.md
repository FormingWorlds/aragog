# Reference data

Aragog's entropy solver requires a directory of pressure-entropy (P-S) equation-of-state tables. This page documents the file format, the canonical PALEOS data source, and the optional external mesh file used when the structure profile is supplied from outside (typically by the Zalmoxis structure solver).

## Pressure-entropy EOS tables

The directory passed to `EntropySolver.from_file(eos_dir=...)` (or referenced through PROTEUS) must contain ten files in the SPIDER-format $(P, S)$ layout:

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
| `solidus_P-S.dat` | 2-column $(P, S)$ | Solidus entropy at each pressure |
| `liquidus_P-S.dat` | 2-column $(P, S)$ | Liquidus entropy at each pressure |

### File layout

The 2D tables follow SPIDER's text format: a five-line header (table title and four metadata lines) followed by a flat list of `P S value` triples on a regular pressure-entropy grid. The pressure axis is in pascals and the entropy axis in joules per kilogram per kelvin. The `solidus_P-S.dat` and `liquidus_P-S.dat` files are two-column $(P, S)$ in the same units, with $P$ monotonically increasing.

The grid must be rectangular: phase-filtered tables (where each phase's grid is restricted to its own valid region) cause `scipy.interpolate.RegularGridInterpolator` to fall back to slow unstructured interpolation and should not be used.

### Canonical source: PALEOS

In the PROTEUS coupled path the tables are produced by the PALEOS multiphase EOS framework ([Attia et al. (2026)](https://scixplorer.org/abs/2026arXiv260503741A/abstract)) from a configured P-T melting curve and the [Wolf & Bower (2018)](https://scixplorer.org/abs/2018PEPI..278...59W/abstract) RTpress liquid EOS. The PROTEUS wrapper writes the resulting tables into `output/data/aragog_pt/` at construction time and points `eos_dir` there.

## External mesh file (`eos_method = 2`)

When the mesh `eos_method` is set to 2, Aragog reads a four-column external file and uses it for the pressure, density, and gravity profiles instead of the analytic Adams-Williamson EOS:

| Column | Unit | Description |
|--------|------|-------------|
| `r` | m | Radial coordinate |
| `P` | Pa | Pressure at that radius |
| `rho` | kg/m³ | Density at that radius |
| `g` | m/s² | Gravitational acceleration at that radius |

The radius column must be monotonically increasing and span at least 75% of the mantle thickness. The PROTEUS wrapper writes this file from the Zalmoxis structure solver into `output/data/zalmoxis_output.dat` (Zalmoxis path) or `output/data/spider_mesh.dat` (SPIDER-static path). When `eos_method = 2` the per-node gravity column overrides the scalar `gravitational_acceleration` config key, so the mixing-length theory and the gravitational-separation flux pick up the radial dependence of $g$ automatically.

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

## Configuration keys at a glance

The relevant configuration keys are summarised here for cross-reference; the full schema is in [How-to: configuration](../How-to/configuration.md).

| Key | Section | Sets |
|-----|---------|------|
| `eos_dir` (constructor argument) | -- | Directory containing the P-S tables |
| `eos_method` | `[mesh]` | `1` = Adams-Williamson; `2` = external file |
| `eos_file` | `[mesh]` | Path to the four-column external mesh file |
| `surface_density`, `adiabatic_bulk_modulus`, `adams_williamson_beta`, `gravitational_acceleration`, `surface_pressure` | `[mesh]` | Adams-Williamson parameters |
