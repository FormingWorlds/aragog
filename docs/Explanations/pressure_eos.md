# Pressure EOS

Aragog uses a *static* pressure equation of state to fix the radial pressure, density, and gravity profile of the mantle mesh on which the entropy solve runs.
This pressure EOS is distinct from the *thermodynamic* $(P, S)$ tables that supply $T$, $\rho$, $c_p$, $\alpha$ at each radial node during the entropy evolution; the thermodynamic layer is documented in [Phase transitions](phase_transitions.md).
The pressure layer fixes $P(r)$, $\rho(r)$, $g(r)$ once at solver setup and is not re-evaluated each RHS call.

Two configurations are available, selected by `mesh.eos_method`:

| `mesh.eos_method` | Class | Use case |
|-------------------|-------|----------|
| `1` | `AdamsWilliamsonEOS` | SPIDER-parity exponential profile derived from surface density and bulk modulus. Default for standalone runs and analytical-mode tests. |
| `2` | `UserDefinedEOS` | External four-column file (`r`, `P`, `rho`, `g`) supplied by Zalmoxis or another structure model. **Default for PROTEUS-coupled runs** (set automatically when the configured structure module is `zalmoxis`, `spider`, or `dummy`). |

Both classes are implemented in [`aragog.mesh.pressure_eos`](../Reference/api/aragog.mesh.md).

## Adams-Williamson EOS (`eos_method = 1`)

The Adams-Williamson EOS assumes hydrostatic balance with a constant adiabatic bulk modulus $K_S$ and gravity $g$.
Combined, these give the closed-form exponential density profile

$$
\rho(z) = \rho_s \exp(\beta z), \qquad \beta = \frac{\rho_s\,g}{K_S},
$$

with $z = R_\mathrm{surf} - r$ the depth and $\rho_s$ the surface density.
The pressure follows by integration of $dP/dr = -g\rho$:

$$
P(r) = \frac{\rho_s\,g}{\beta}\left(e^{\beta z} - 1\right) + P_\mathrm{surf},
$$

which is invertible analytically:

$$
r(P) = R_\mathrm{surf} - \frac{1}{\beta}\,\ln\!\left(1 + \frac{(P - P_\mathrm{surf})\,\beta}{\rho_s\,g}\right).
$$

The solver uses the analytic inverse directly; no Newton iteration is required for $r(P)$.

This matches SPIDER's [`eos_adamswilliamson.c`](https://github.com/FormingWorlds/SPIDER) line by line.
The previous Aragog implementation used a rational (hyperbolic) form $\rho = \rho_s K_S / (K_S - \rho_s g z)$ derived from the same $K_S$ but differing from SPIDER by up to 6 % at CMB depth, causing 3 % $R_\mathrm{int}$ mismatches on the same target planet mass.
Switching to the exponential form closed that gap.

### Beta override

`mesh.adams_williamson_beta` lets the user pass $\beta$ directly.
When the override is positive, it takes precedence over the $K_S$-derived value; when zero (default), the EOS computes $\beta = \rho_s g / K_S$.
The override is the normal path for SPIDER-parity runs because SPIDER reads $\beta$ from its own input file.

### When the closed form is wrong

The constant-$K_S$ assumption breaks down for super-Earths and sub-Neptunes where the bulk modulus varies by an order of magnitude across the mantle.
For those planets, `eos_method = 2` with a Zalmoxis-exported PALEOS profile is the correct choice and is the production-grade path inside PROTEUS.

## PALEOS via Zalmoxis (`eos_method = 2`)

The user-defined EOS path reads a four-column whitespace-separated file with columns `radius`, `pressure`, `density`, `gravity` and interpolates each column onto the basic-node radii.
The intended source for this file in production runs is [Zalmoxis](https://proteus-framework.org/Zalmoxis/), the PROTEUS structure solver, which couples to the [PALEOS](https://scixplorer.org/abs/2026arXiv260503741A/abstract) multiphase equation of state ([Attia et al. (2026)](https://scixplorer.org/abs/2026arXiv260503741A/abstract)).

### What PALEOS provides

PALEOS is a multiphase EOS framework for rocky-planet interiors that couples thermal, structural, and phase-state information across the mantle.
For a given bulk composition, target mass, and surface boundary condition, PALEOS solves the hydrostatic-equilibrium structure self-consistently with the chosen EOS family (e.g. RTpress, Birch-Murnaghan, polynomial fits to ab initio data).
The output is a fully consistent ($r$, $P$, $\rho$, $g$, $T$, $\phi$) profile down to the core-mantle boundary, with the bulk modulus, thermal expansivity, and other derived quantities all evaluated from the same EOS family at every radial node.
PALEOS's main feature relative to a one-shot Adams-Williamson profile is that it covers super-Earth and sub-Neptune mass ranges where the bulk modulus varies by more than an order of magnitude across the mantle.

### Standalone use

For standalone runs, the user supplies the four-column file directly:

```toml
[mesh]
eos_method = 2
eos_file = "/path/to/your/profile.dat"  # four columns: r [m], P [Pa], rho [kg/m^3], g [m/s^2]
```

Any provenance for this file is acceptable, as long as the four columns are physically consistent.
A PALEOS-derived export is the recommended source in the PROTEUS ecosystem because the same EOS family is then used for the structural solve, the thermodynamic $(P, S)$ tables, and (downstream) the magma-ocean energy balance, which keeps the three layers self-consistent.
Aragog is otherwise EOS-agnostic at this layer: any file with monotone radii and physically consistent columns works.

### PROTEUS-coupled use (default production path)

When PROTEUS drives Aragog, the wrapper at `proteus.interior_energetics.aragog` overrides `mesh.eos_method = 2` and points `mesh.eos_file` at the structure-solver export:

| Configured `interior_struct.module` | File written | EOS source |
|-------------------------------------|--------------|------------|
| `zalmoxis` (production default) | `data/zalmoxis_output.dat` | PALEOS, full multiphase EOS |
| `spider` or `dummy` | `data/spider_mesh.dat` | Adams-Williamson at the SPIDER-parity setting (when present) |

For Zalmoxis-coupled runs, this means:

1. Zalmoxis runs first at each coupling step (or at simulation init, depending on how often the structure is refreshed).
2. Zalmoxis solves the planet's hydrostatic-equilibrium structure with PALEOS as the EOS, giving a self-consistent $(r, P, \rho, g)$ profile to the configured target mass and composition.
3. Zalmoxis writes the profile to `data/zalmoxis_output.dat`.
4. The Aragog wrapper sets `mesh.eos_method = 2` and `mesh.eos_file = data/zalmoxis_output.dat`.
5. Aragog interpolates the file onto its basic-node radii and uses the resulting $P(r)$, $\rho(r)$, $g(r)$ profiles for the entropy solve.

The thermodynamic $(P, S)$ tables that Aragog uses for the property lookups during the entropy evolution are also PALEOS-derived in this configuration; see [Phase transitions](phase_transitions.md) for that layer.

### Interpolation and validation

Aragog interpolates each of the four columns onto the basic-node radii using `scipy.interpolate.PchipInterpolator` (shape-preserving cubic).
The cumulative mass is precomputed via `scipy.integrate.cumulative_trapezoid` and then PCHIP-interpolated as well, so `get_mass_within_radii` is $O(1)$ per call.

`EntropySolver._validate_eos_radius_range` (in `solver/entropy_solver.py`) asserts that the file's radius array is strictly monotonic and that its span covers at least 75 % of the mesh's $[R_\mathrm{cmb}, R_\mathrm{surf}]$ range, with a relative tolerance scaled to the mesh span (and a 1 m absolute floor) so that single-ULP drift on a Zalmoxis-resumed mesh does not trip the check.
Otherwise a `ValueError` is raised.
This guards against an undersized external mesh silently clamping or extrapolating the deep interior.

## Mass coordinates

When `mesh.mass_coordinates = true` the basic-node radii are spaced uniformly in mass coordinate $\xi$ rather than radius $r$.
The mass coordinate is

$$
\xi(r) = \frac{m(r) - m(R_\mathrm{cmb})}{m(R_\mathrm{surf}) - m(R_\mathrm{cmb})}
$$

with $m(r)$ the cumulative mass enclosed by radius $r$.
For the Adams-Williamson EOS the antiderivative is closed-form:

$$
M(r) = 4\pi \rho(r) \left( -\frac{2}{\beta^3} - \frac{r^2}{\beta} - \frac{2r}{\beta^2} \right) + C,
$$

matching SPIDER's `eos_adamswilliamson.c` line 191.
The constant $C$ cancels in any difference $M(r_2) - M(r_1)$, so anchoring at $r_\mathrm{cmb}$ is sufficient.

For uniformly spaced $\xi$, the corresponding $r$ values are found by inverting $M(r)$ via `scipy.optimize.brentq`, bracketed between the surface and CMB.
brentq is preferred over Newton because $M(r)$ is strictly monotonic and the bracketed root is guaranteed.

For the user-defined path (`eos_method = 2`) the cumulative mass is built from the supplied $\rho(r)$ via `cumulative_trapezoid`, then PCHIP-interpolated; the same brentq inversion gives uniformly $\xi$-spaced radii.

## Picking a configuration

| Goal | Recommended `eos_method` |
|---|---|
| Earth-mass standalone smoke test | `1` (Adams-Williamson) |
| SPIDER-parity reproduction | `1` with `adams_williamson_beta` set explicitly |
| Super-Earth or sub-Neptune mantle | `2` with a PALEOS-derived file |
| PROTEUS-coupled production run | **`2` (set automatically by the wrapper from the Zalmoxis structure)** |
| Constant-properties analytical mode | `1` (suffices; the analytical path bypasses the table lookups) |

For a fuller treatment of when PALEOS-driven structures matter and how the standalone vs PROTEUS-coupled paths share the same numerical core, see [Standalone usage](../How-to/usage-paths.md) and [Coupling Aragog to PROTEUS](../How-to/proteus_coupling.md).
