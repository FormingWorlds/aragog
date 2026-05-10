# Verification figures

Reproducible scripts that generate the seven first-principles verification figures embedded in `docs/Explanations/verification.md`.
Each script is a self-contained verification of one numerical or physical aspect of Aragog and writes its output to `docs/figures/vv/<stem>.{pdf,png}` (tracked in git) plus a raw-data `.npz` to `output/aragog_vv_data/` (gitignored).

## Layout

| Script | Verifies | Output figure |
|---|---|---|
| `verify_eos_bilinear_jacobian.py` | JAX-traced bilinear interpolation of the EOS table reproduces the numpy reference and supplies finite, non-zero derivatives via `jax.jacrev`. | `fig_03_eos_bilinear_jacobian` |
| `verify_flux_decomposition.py` | Per-cell heat flux components ($F_\text{cond}$, $F_\text{conv}$, $F_\text{grav}$, $F_\text{mix}$) reconstruct the total heat flux to floating-point round-off. | `fig_02_flux_decomposition` |
| `verify_mass_coord_jacobian.py` | The mesh-internal mass-coordinate transform Jacobian $d\xi/dr$ matches the analytic form $(\rho^* / \rho^*_\mathrm{planet})\,(r/\xi)^2$. | `fig_07_mass_coord_jacobian` |
| `verify_permeability.py` | Three-regime gravitational-separation permeability $F(\zeta)$ implements the Abe (1993, 1995) Stokes / Rumpf-Gupte / Blake-Kozeny-Carman blend with the documented critical porosities. | `fig_04_permeability` |
| `verify_radio_decay.py` | `aragog.jax.solver.make_radio_heating_fn` reproduces the analytical exponential-decay law for the four long-lived isotopes plus Al-26 and Fe-60 at canonical Solar-System-initial abundances. | `fig_06_radio_decay` |
| `verify_rhs_parity.py` | The JAX RHS reproduces the numpy RHS (component-wise and assembled $dS/dt$) at machine precision across multiple representative entropy states. | `fig_01_rhs_parity` |
| `verify_utbl_cardano.py` | The closed-form Cardano root used by the upper-thermal-boundary-layer correction recovers $T_\mathrm{surf} < T_\mathrm{int}$ across the physical range. | `fig_05_utbl_cardano` |

`_style.py` is a shared style helper (color palette, panel labels, save helper). It is not invoked directly.

## Running

All seven scripts depend on the production aragog environment (numpy, JAX, equinox, scipy; the JAX path uses `jax.config.update('jax_enable_x64', True)` so float64 is required):

```sh
conda activate proteus    # or your aragog env

# Self-contained scripts (no external fixture needed):
python tools/verification/figures/verify_eos_bilinear_jacobian.py
python tools/verification/figures/verify_permeability.py
python tools/verification/figures/verify_radio_decay.py
python tools/verification/figures/verify_utbl_cardano.py
```

Three scripts (`verify_rhs_parity.py`, `verify_flux_decomposition.py`, `verify_mass_coord_jacobian.py`) build a coupled Aragog/JAX solver fixture from a PROTEUS-side scaffolding script. Set `PROTEUS_SCRIPTS` to the directory holding the fixture before running:

```sh
export PROTEUS_SCRIPTS=$HOME/git/PROTEUS/scripts
python tools/verification/figures/verify_rhs_parity.py
python tools/verification/figures/verify_flux_decomposition.py
python tools/verification/figures/verify_mass_coord_jacobian.py
```

If `PROTEUS_SCRIPTS` is unset the three fixture-dependent scripts will raise an `ImportError` at the call site that imports `z02_parity_multi_state`. Replacing this with a portable bundled-config fixture is on the follow-up list.

## Outputs

- Figures: `docs/figures/vv/fig_NN_<topic>.pdf` and `.png` (both tracked in git so the docs site builds without local script execution).
- Raw NumPy data: `output/aragog_vv_data/fig_NN_<topic>.npz`. Gitignored; regenerated each time a script runs.
