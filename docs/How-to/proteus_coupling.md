# Coupling Aragog to PROTEUS

This page is the practical recipe for running Aragog as the interior-energetics module inside the PROTEUS framework.
The standalone Python API described in [Standalone usage](usage-paths.md) does *not* apply when Aragog is driven from PROTEUS.
PROTEUS bypasses Aragog's TOML loader entirely and builds the call-time configuration from its own `[interior_energetics]` and `[interior_energetics.aragog]` blocks.

For the theory behind this design, see [Coupling to PROTEUS (theory)](../Explanations/proteus_coupling.md).

!!! info "Audience"
    This guide assumes you already have a working PROTEUS install (see the [PROTEUS docs](https://proteus-framework.org/PROTEUS)) and a coupled atmosphere + interior + outgassing config.
    If you are running Aragog on its own, ignore this page and follow [Standalone usage](usage-paths.md).

---

## Minimal coupling block

In a PROTEUS TOML config, set Aragog as the energetics module with:

```toml
[interior_energetics]
module           = "aragog"
num_levels       = 80
rtol             = 1e-10
atol             = 1e-10
trans_conduction = true
trans_convection = true
trans_grav_sep   = true
trans_mixing     = true
heat_radiogenic  = true

[interior_energetics.aragog]
backend                     = "jax"          # CVODE + JAX analytic Jacobian
core_bc                     = "energy_balance"   # SPIDER bit-parity
solver_method               = "cvode"
mass_coordinates            = true
atol_temperature_equivalent = 1e-8
phi_step_cap                = 0.05
```

This is the production-default pairing: SUNDIALS CVODE with the JAX-derived analytic Jacobian, the `energy_balance` core BC for SPIDER bit-parity, mass coordinates for SPIDER-equivalent meshing, and a per-call $|\Delta\Phi|$ cap of 0.05.

!!! tip "EOS table provenance"
    When `interior_struct.module = "zalmoxis"`, Aragog's P-T tables are written by Zalmoxis on the fly from the active PALEOS EOS into `<outdir>/data/aragog_pt/`.
    Aragog reads them through its non-uniform grid loader; no manual pre-processing is needed.
    Phase-filtering the input PALEOS table breaks scipy's `RegularGridInterpolator`; the Zalmoxis-side generator writes the full rectangle, and you should not edit it between write and read.

---

## Backend selection

`backend = "jax"` (default) routes the RHS through `aragog.jax` and the analytic Jacobian through `jax.jacrev`. The JAX path eliminates the noise of CVODE's finite-difference Jacobian at production tolerances; it is the only path that consistently survives the rheological transition without tripping Aragog's T_core-jump retry guard.

`backend = "numpy"` is supported for development, SPIDER-parity comparisons, and machines without a JAX-compatible build. The numpy RHS feeds CVODE with a finite-difference Jacobian. Functionally correct, but slower per call and less robust at tight tolerances.

```toml
[interior_energetics.aragog]
backend = "jax"     # production (default)
# backend = "numpy" # development / parity comparisons only
```

The `diffrax` direct-JAX integration path is research-only and is not exposed through the schema.

---

## Core boundary condition

Aragog supports four core-mantle boundary modes:

| Mode | When to use | Notes |
|---|---|---|
| `"energy_balance"` (default) | Production. SPIDER bit-parity. | Implements the [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) `bc.c::core_BC` formula. Adds `dSdr_cmb` as a state variable; conserves core enthalpy across the CMB. |
| `"quasi_steady"` | SPIDER-legacy parity tests only. | Legacy v3 alpha-factor heat-flux partition. Gives a $\sim 19$% T_core offset against SPIDER on a typical Earth IC; do not use for new runs. |
| `"gradient"` | Numerical-experiment use. | Two boundary entropies promoted to state variables. |
| `"bower2018"` | Experimental. | Do not use for production. |

```toml
[interior_energetics.aragog]
core_bc = "energy_balance"
```

The `energy_balance` mode requires `mass_coordinates = true` to avoid a basal-cell volume mismatch on the SPIDER mesh. Both flags default to production-correct values; do not change one without the other.

---

## Per-call melt-fraction cap (`phi_step_cap`)

Aragog ships a SUNDIALS root function that caps the per-call $|\Delta\Phi_\mathrm{global}|$ excursion. When `phi_step_cap > 0`, CVODE evaluates

$$
g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t, y) - \Phi_\mathrm{global}(t_\mathrm{start})|
$$

at every internal step and returns at the exact zero-crossing (`flag = CV_ROOT_RETURN`). The integrator's adaptive step never overshoots a physically meaningful melt-fraction excursion.

```toml
[interior_energetics.aragog]
phi_step_cap = 0.05      # production default
# phi_step_cap = 0.0     # disabled; native CVODE step control
# phi_step_cap = 0.001   # very tight; only when oscillations show up early
```

With `phi_step_cap = 0.0` the rootfn is not registered. The scipy fallbacks (`solver_method = "radau"` or `"bdf"`) register the equivalent capping logic as a `solve_ivp` event with `terminal=True`, so the same TOML behaves consistently across backends.

See the dedicated [`phi_step_cap` how-to](phi-step-cap.md) for tuning guidance.

---

## Tolerances

Tolerances are set in two places: `interior_energetics.{rtol, atol}` (the integrator side) and `interior_energetics.aragog.atol_temperature_equivalent` (the temperature-scaled absolute tolerance Aragog converts internally to entropy units via $C_p / T$).

```toml
[interior_energetics]
rtol = 1e-10
atol = 1e-10

[interior_energetics.aragog]
atol_temperature_equivalent = 1e-8
```

The `1e-8` temperature-equivalent default mirrors SPIDER's `atol = rtol = 1e-8` setting and eliminates the CVODE marginal-stability bifurcation that appeared at the first dt jump after equilibration with a looser `3e-7`. Tightening further has diminishing returns; loosening risks silent integration drift across the rheological transition.

---

## Energy-transport switches

Aragog's RHS is the sum of four transport terms plus radiogenic heating. Each is gated by a boolean flag:

```toml
[interior_energetics]
trans_conduction = true     # solid conduction
trans_convection = true     # mixing-length convection
trans_mixing     = true     # composition mixing flux (Δh-weighted)
trans_grav_sep   = true     # gravitational separation flux (Δh-weighted)
heat_radiogenic  = true     # six-isotope Ruedas (2017) cocktail
heat_tidal       = false    # off by default
```

The validator at `proteus.config._interior.valid_aragog` rejects configs that disable all four transport terms. The radiogenic switch toggles whether the Ruedas (2017) six-isotope cocktail (²⁶Al, ⁴⁰K, ⁶⁰Fe, ²³²Th, ²³⁵U, ²³⁸U) feeds into the entropy source term.

---

## Radiogenic heating

When `heat_radiogenic = true`, Aragog reads abundances from the four PROTEUS-side keys:

```toml
[interior_energetics]
radio_tref = 4.567      # reference age [Gyr]; Earth formation (matches the schema default in proteus.config._interior)
radio_K    = 310        # K-40 concentration at t = radio_tref [ppmw]
radio_U    = 0.031      # U (²³⁵+²³⁸) [ppmw]
radio_Th   = 0.124      # Th-232 [ppmw]
```

Aluminium-26 and iron-60 are exposed at the Aragog-side `cfg/` defaults (zero-concentration entries that activate only if you override them). For PROTEUS coupled runs, the four-isotope $K/U/U/Th$ subset is sufficient unless you are modelling a $< 5$ Myr post-formation epoch where ²⁶Al matters.

The decay constants and specific-heat-production rates come from [Ruedas (2017)](https://scixplorer.org/abs/2017GGG....18.3530R/abstract).

---

## External mesh handover (Zalmoxis)

When `interior_struct.module = "zalmoxis"`, Zalmoxis writes a five-column TSV (`r, P, rho, g, T`) to `<outdir>/data/zalmoxis_output.dat` on every successful re-solve. Aragog reads it inside `solver.reset()` to rebuild its mass-coordinate mesh; only the first four columns are consumed, the temperature column is a Zalmoxis diagnostic and is ignored on the Aragog side.

The handover is enforced by a schema check in `proteus.interior_struct.zalmoxis.validate_zalmoxis_output_schema`: top-of-mantle radius equality to relative tolerance $10^{-6}$, and mantle mass conservation to $5 \times 10^{-2}$. Per-call radius shifts above `interior_struct.zalmoxis.mesh_max_shift` are blended in `proteus.interior_energetics.spider.blend_mesh_files` (shared between the Aragog and SPIDER paths) to keep the mesh transition trackable.

```toml
[interior_struct]
module          = "zalmoxis"
core_density    = "self"
core_heatcap    = "self"

[interior_struct.zalmoxis]
mesh_max_shift  = 0.05
```

For the full coupling theory see the [Coupling theory page](../Explanations/proteus_coupling.md#external-mesh-handover).

---

## Worked example: 1 $M_\oplus$ PROTEUS + Zalmoxis + Aragog

```toml
[planet]
mass_tot           = 1.0
temperature_mode   = "adiabatic_from_cmb"
tcmb_init          = 7199.0
tsurf_init         = 3830.0
tcenter_init       = 6000.0
ini_entropy        = 3900.0
prevent_warming    = false

[interior_struct]
module          = "zalmoxis"
core_frac       = 0.325
core_frac_mode  = "mass"
core_density    = "self"
core_heatcap    = "self"

[interior_struct.zalmoxis]
core_eos             = "PALEOS:iron"
mantle_eos           = "PALEOS-2phase:MgSiO3"
mushy_zone_factor    = 0.8
mantle_mass_fraction = 0.0
num_levels           = 150
outer_solver         = "newton"
update_interval      = 50000.0
update_dphi_abs      = 0.05
update_dtmagma_frac  = 0.05
mesh_max_shift       = 0.05
equilibrate_init     = true
dry_mantle           = true

[interior_energetics]
module           = "aragog"
num_levels       = 80
rtol             = 1e-10
atol             = 1e-10
trans_conduction = true
trans_convection = true
trans_grav_sep   = true
trans_mixing     = true
heat_radiogenic  = true

[interior_energetics.aragog]
backend                     = "jax"
core_bc                     = "energy_balance"
solver_method               = "cvode"
mass_coordinates            = true
atol_temperature_equivalent = 1e-8
phi_step_cap                = 0.05
```

Run with:

```bash
nohup proteus start -c <this.toml> --offline > output/<run>/launch.log 2>&1 & disown
```

For numerically fragile anchors (wet 1 $M_\oplus$ at IW+4, reduced 1 $M_\oplus$ at IW-2), add `--deterministic`.

---

## Prioritised settings

The list below is the short version of "what actually matters" when you stand up a new coupled Aragog run.
Knobs are ordered from highest to lowest impact on results and stability.
Every default is read directly from `proteus.config._interior` (see `Aragog` and `Interior`); if a quoted default disagrees with the source, the source wins.

### 1. `interior_energetics.module`

**Default**: `"aragog"`. **Recommendation**: keep `"aragog"` for new production runs.

Pair Aragog with Zalmoxis (`interior_struct.module = "zalmoxis"`) for the production combination. The alternative `"spider"` energetics module is supported for parity comparisons but requires `core_frac_mode = "radius"` because SPIDER does not accept mass-mode core fractions.

### 2. `interior_energetics.aragog.backend`

**Default**: `"jax"`. **Recommendation**: stay on `"jax"`.

The JAX-derived analytic Jacobian is the only path that consistently survives the rheological transition at production tolerances. The numpy path uses CVODE's finite-difference Jacobian and is reserved for development and SPIDER-parity tests.

### 3. `interior_energetics.aragog.core_bc`

**Default**: `"energy_balance"`. **Recommendation**: keep `"energy_balance"`.

Implements [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) `bc.c::core_BC` to bit-parity. The legacy `"quasi_steady"` mode gives a $\sim 19$% T_core offset and is preserved only for SPIDER-legacy parity reproduction.

### 4. `interior_energetics.aragog.phi_step_cap`

**Default**: `0.0` (cap disabled). **Recommendation**: `0.05` for standard evolutionary runs.

The SUNDIALS root function caps per-call $|\Delta\Phi|$ excursions and stops adaptive step overshoot at the rheological transition. Tighten to `0.001` to `0.01` only if mushy-zone oscillations appear in the first few snapshots.

### 5. `planet.prevent_warming`

**Default**: `false`. **Recommendation**: leave at `false`.

When set to `true`, all atmosphere modules and termination checks enforce a `T_magma = min(T_new, T_prev)` clamp. This is energy-non-conserving in any regime where the integrator transiently warms the magma, producing an apparent T_magma plateau with an energy-leak signature. All bundled coupled configs default to `prevent_warming = false`.

### 6. `interior_energetics.{rtol, atol}` and `interior_energetics.aragog.atol_temperature_equivalent`

**Defaults**: `rtol = 1e-10`, `atol = 1e-10`, `atol_temperature_equivalent = 1e-8`.
**Recommendation**: keep production defaults.

Loosening to `1e-7` re-introduces the CVODE marginal-stability bifurcation at iter $\sim 9$ and silent drift across the rheological transition. Tightening below `1e-10` has diminishing returns.

### 7. `interior_energetics.num_levels`

**Default**: `80`. **Recommendation**: keep at `80`.

This matches the SPIDER reference and keeps CVODE matrix factorisations cheap. Halving does not save wall time linearly because the dominant cost is per-cell EOS evaluation; raising rarely pays off.

### 8. `interior_energetics.aragog.mass_coordinates`

**Default**: `true`. **Recommendation**: keep `true`.

Uniform spacing in mass-coordinate space gives larger cells at the surface where density is lower, matching SPIDER's mesh. Required for the `energy_balance` core BC; do not change one without the other.

#### Summary table

| Knob | Default | Recommended | Why it matters |
|---|---|---|---|
| `interior_energetics.module` | `"aragog"` | `"aragog"` | CVODE + JAX analytic Jacobian, robust at production tolerances. |
| `interior_energetics.aragog.backend` | `"jax"` | `"jax"` | Eliminates FD-Jacobian noise at the rheological transition. |
| `interior_energetics.aragog.core_bc` | `"energy_balance"` | `"energy_balance"` | SPIDER bit-parity; correct T_core. |
| `interior_energetics.aragog.phi_step_cap` | `0.0` | `0.05` | SUNDIALS rootfn caps per-call $|\Delta\Phi|$ excursion. |
| `planet.prevent_warming` | `false` | `false` | The clamp is energy-non-conserving in warming sub-steps. |
| `interior_energetics.rtol` / `atol` | `1e-10` | `1e-10` | Schema default; avoids CVODE marginal-stability bifurcation. Production CHILI runs occasionally relax `rtol` to `1e-8`; do not loosen further. |
| `interior_energetics.aragog.atol_temperature_equivalent` | `1e-8` | `1e-8` | Matches SPIDER `atol = rtol = 1e-8`. |
| `interior_energetics.num_levels` | `80` | `80` | SPIDER reference resolution. |
| `interior_energetics.aragog.mass_coordinates` | `true` | `true` | Required by `energy_balance` core BC. |

---

## Common pitfalls

- **Setting an Aragog-side TOML in a PROTEUS config does nothing.** Aragog's TOML loader is the standalone path; PROTEUS reads its own `[interior_energetics.aragog]` block. The two paths are mutually exclusive.
- **`core_bc = "energy_balance"` without `mass_coordinates = true`** is a silent basal-cell volume mismatch. The schema does not yet enforce the pairing; keep both at production defaults.
- **`phi_step_cap = 0.0` plus a hot wet super-Earth IC** can hit a CVODE step that overshoots the rheological transition. Switch to `0.05`.
- **Disabling all four `trans_*` flags** is rejected by `valid_aragog`. At least one transport term must be active.
- **`backend = "numpy"` at production tolerances** can trip Aragog's T_core-jump retry guard at the first dt jump. Switch to `"jax"` (or loosen tolerances explicitly, knowing this will silently drift across the rheological transition).

For the why behind these, see the [Coupling to PROTEUS theory page](../Explanations/proteus_coupling.md).
