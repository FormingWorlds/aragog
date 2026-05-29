---
title: Aragog in the PROTEUS framework
---

<h1 align="center">
    <a href="https://proteus-framework.org">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black_nobkg.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
    </a>
</h1>


Aragog is the **interior energetics module** of [PROTEUS](https://proteus-framework.org/PROTEUS) (/ˈproʊtiəs/, PROH-tee-əs), a modular Python framework for the coupled evolution of the atmospheres and interiors of rocky planets and exoplanets. A schematic of PROTEUS components and corresponding modules can be found below. Click any module in the diagram to open its documentation, or navigate to it from the sidebar.
<br>

<object type="image/svg+xml" data="https://cdn.jsdelivr.net/gh/FormingWorlds/PROTEUS@main/docs/assets/proteus_modules_schematic.svg" class="mod-diagram mod-diagram--light" aria-label="PROTEUS module schematic (light mode)">PROTEUS module schematic (light mode)</object>
<object type="image/svg+xml" data="https://cdn.jsdelivr.net/gh/FormingWorlds/PROTEUS@main/docs/assets/proteus_modules_schematic_darkmode.svg" class="mod-diagram mod-diagram--dark" aria-label="PROTEUS module schematic (dark mode)">PROTEUS module schematic (dark mode)</object>

<p style="text-align: center;"><strong>Schematic of PROTEUS components and corresponding modules.</strong></p>

---

## Where Aragog sits in a coupled PROTEUS run

In a PROTEUS coupled run, four submodule families deliver the mantle-atmosphere physics in a fixed per-iteration order:

| Submodule | Role | State variables produced |
|---|---|---|
| **Outgassing** (CALLIOPE / Atmodeller) | Volatile partitioning between magma ocean and atmosphere | $X_i^\mathrm{melt}$, $X_i^\mathrm{atm}$ |
| **Interior structure** (Zalmoxis, optionally SPIDER or `dummy`) | Static structure: hydrostatic equilibrium, mass-radius, density, gravity, $R_\mathrm{cmb}$, $P_\mathrm{cmb}$ | $\rho(r)$, $g(r)$, $P(r)$, mesh file |
| **Interior energetics** (**Aragog**, SPIDER, or `dummy`) | Thermal evolution: entropy ODE, $T(r)$ trajectory, surface heat flux | $S(r)$, $T(r)$, $\Phi$, $F_\mathrm{atm}$, $T_\mathrm{magma}$, $T_\mathrm{core}$ |
| **Atmosphere** (AGNI / JANUS / dummy) | Radiative-balance surface temperature | $T_\mathrm{surf}$, atmospheric column |

Aragog reads $\rho(r)$, $g(r)$, and $P(r)$ from the mesh file written by the interior-structure step (`data/zalmoxis_output.dat` for Zalmoxis-coupled runs), reads the surface boundary condition from the atmosphere step, and writes the entropy profile and a small set of scalar diagnostics to `hf_row` for the next iteration.
Outgassing and the atmosphere step both consume `T_magma` and the surface flux that Aragog produces.

The per-iteration cadence and the conditions under which Zalmoxis re-solves (rather than reusing the previous mesh) are described in [Coupling to PROTEUS (theory)](Explanations/proteus_coupling.md).

---

## Quick links into the coupling docs

The two pages dedicated to the PROTEUS coupling on the Aragog side are:

<div class="grid cards" markdown>

-   :material-rocket-launch: **How to couple Aragog to PROTEUS**

    Practical TOML recipe: minimal `[interior_energetics]` block, the `[interior_energetics.aragog]` knobs in priority order, the JAX + CVODE production path, the `phi_step_cap` rootfn for crossing the rheological transition, the retry ladder, and the common pitfalls.

    [Go to the how-to](How-to/proteus_coupling.md)

-   :material-book-open-variant: **Theory of the coupling**

    Per-iteration call sequence, the `hf_row` keys consumed and produced, the `EntropySolver` lifecycle (init, equilibrate, evolve, reset on Zalmoxis re-solve), the JAX CVODE factory registration, and the design rationale for keeping the coupling at the function-call level rather than via standalone TOML.

    [Go to the explainer](Explanations/proteus_coupling.md)

</div>

For the broader theoretical context that motivates Aragog inside PROTEUS, see also:

- [Mixing-length theory](Explanations/mixing_length.md): the local convective closure that Aragog uses, contrasted with the boundary-layer-theory closure used by the lump-magma-ocean lineage.
- [Two-phase flow in Aragog](Explanations/two_phase_flow.md): the solid + melt mixture treatment that activates gravitational separation, chemical mixing, and continuous lever-rule latent heat.
- [Pressure EOS](Explanations/pressure_eos.md): the static pressure-and-density profile, with `mesh.eos_method = 2` (Zalmoxis-driven, PALEOS-based) as the PROTEUS default.
- [Phase transitions](Explanations/phase_transitions.md): the thermodynamic $(P, S)$ tables that supply $T$, $\rho$, $c_p$, $\alpha$ at every radial node.
- [Aragog vs SPIDER](Explanations/spider_comparison.md): what SPIDER does the same and where Aragog deliberately diverges.

For the PROTEUS-side documentation (config schema, orchestrator behaviour, atmosphere modules, outgassing modules), see [proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS).

---

## Standalone vs PROTEUS-coupled

Aragog is also a fully self-contained tool for standalone interior dynamics simulations and parameter studies.
The two modes share the same numerical core but are configured at different layers:

| Mode | Entry point | Configuration |
|---|---|---|
| **Standalone** | `EntropySolver(parameters)` from a Python script, or `aragog` CLI | TOML config files (`abe_mixed.toml`, `abe_solid.toml`, etc.) with `[mesh]`, `[energy]`, `[boundary_conditions]`, `[phase_solid]`, `[phase_liquid]`, `[phase_mixed]`, `[radionuclides]`, `[solver]` sections. Documented under [Standalone usage](How-to/usage-paths.md) and [Configuration](How-to/configuration.md). |
| **PROTEUS-coupled** | `proteus start -c <run>.toml` | PROTEUS-side TOML sections `[interior_energetics]` and `[interior_energetics.aragog]`, plus the relevant `[interior_struct]`, `[outgas]`, `[atmos]` blocks. The Aragog-side TOML sections above are not read; the wrapper builds `Parameters` directly. Documented in [Coupling to PROTEUS](How-to/proteus_coupling.md). |

The Aragog API, the JAX path, and the SUNDIALS CVODE integrator are identical in the two modes; only the TOML scaffolding around them differs.
For a side-by-side comparison see [Standalone usage](How-to/usage-paths.md).

---

## Where the wrapper code lives

Aragog is invoked from PROTEUS via:

- `proteus/interior_energetics/aragog.py`: the wrapper that builds `Parameters`, instantiates `EntropySolver`, registers the JAX CVODE factory, drives the per-iteration `solve()` calls, manages the retry ladder for Zalmoxis-coupled re-solves, and translates the resulting `SolverOutput` into `hf_row` keys.
- `proteus/config/_interior.py`: the attrs-based schema for `[interior_energetics]` and `[interior_energetics.aragog]`. All knobs documented in the how-to map back to fields here.
- `proteus/interior_energetics/wrapper.py`: hosts `equilibrate_initial_state()`, the pre-main-loop CALLIOPE + Zalmoxis convergence loop that sets the surface temperature and volatile partitioning before Aragog's first thermal solve.

These files live in the [PROTEUS repository](https://github.com/FormingWorlds/PROTEUS), not in Aragog.
Their per-symbol API documentation is rendered in the PROTEUS docs; this site documents the Aragog side of the contract.
