"""V&V Figure 2: Heat-flux decomposition on a CHILI-relevant magma-ocean state.

Decomposes the total radial heat flux F_tot = F_cond + F_conv + F_grav
+ F_mix (the four heat-flux components) and the
internal volumetric heat source H = H_radio + H_dil + H_tidal
(the entropy balance) at every basic node of an Earth-like 80-cell mesh
(``chili_repro_v2.toml``).

Each component is isolated by calling ``aragog.jax.phase.compute_fluxes``
with the corresponding transport flag set to True and all the others to
False. The total is computed by enabling all flags simultaneously. The
sum of the four isolated fluxes must reconstruct the total to within
floating-point round-off (verified in the printed report).

The state used is a fully-mushy profile placed at the midpoint
between the local solidus and liquidus entropies on each cell, with
a small superimposed entropy gradient to drive convection. This is
the regime where every flux component has a non-zero contribution:
gravitational separation and chemical mixing turn on inside the
mushy band, MLT convection is active where the entropy gradient is
unstable, conduction dominates near the boundaries, and dilatation
PdV heating (the entropy balance) develops where the mass fluxes are large.

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_flux_decomposition.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for var in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(var, '1')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent.parent  # tools/verification/figures/ -> repo root
sys.path.insert(0, str(ROOT))
PROTEUS_SCRIPTS = os.environ.get('PROTEUS_SCRIPTS')
if PROTEUS_SCRIPTS:
    sys.path.insert(0, str(Path(PROTEUS_SCRIPTS).resolve()))
from _style import PALETTE, apply_rc, panel_label, save  # noqa: E402

OUT = REPO_ROOT / 'docs' / 'figures' / 'vv'
DATA = REPO_ROOT / 'output' / 'aragog_vv_data'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Production-CHILI radionuclide cocktail (abe_mixed_init.cfg, Ruedas 2017
# + Turcotte & Schubert 2014). Used here only to add a non-zero radio
# contribution to the heat-source bar in panel (b).
RADIO = dict(
    hp = np.array([2.8761e-5, 2.6368e-5, 5.68402e-4, 9.4946e-5]),
    ab = np.array([1.1668e-4, 1.0,        7.2045e-3, 0.9927955]),
    cc = np.array([310.0e-6, 0.124e-6, 0.031e-6, 0.031e-6]),
    t0 = np.array([4.55e9] * 4),
    hl = np.array([1.248e9, 14.0e9, 0.704e9, 4.468e9]),
)


def main():
    apply_rc()
    from z02_parity_multi_state import build_solver_and_jax_args  # noqa: WPS433
    from aragog.jax.phase import PhaseParams, compute_fluxes
    from aragog.jax.solver import make_radio_heating_fn

    solver, args, eos_jax = build_solver_and_jax_args()
    eos, params0, mesh, bc, heating_static = args
    n_stag = solver._n_stag
    r_basic = np.asarray(mesh.radii_basic)
    P_basic = np.asarray(mesh.P_basic)

    # Mushy-band state: midpoint between solidus and liquidus on each
    # cell, plus a small entropy increment from CMB to surface so the
    # convective flux is also non-zero. Tabulated solidus/liquidus from
    # the production EOS is in S~1200-2600 J/kg/K range.
    P_stag = mesh.P_stag
    S_sol = eos.solidus_entropy(P_stag)
    S_liq = eos.liquidus_entropy(P_stag)
    S_mid_phi = 0.5 * (S_sol + S_liq)        # phi ~ 0.5 everywhere
    delta_S = jnp.linspace(0.0, 60.0, n_stag)  # surface 60 J/kg/K hotter
    S_stag = S_mid_phi + delta_S

    # Static radio (uniform per cell). We add it manually below to the
    # H_dil decomposition; compute_fluxes does NOT mix radio in.
    H_radio_fn = make_radio_heating_fn(
        RADIO['hp'], RADIO['ab'], RADIO['cc'], RADIO['t0'], RADIO['hl'],
    )
    H_radio_per_cell = float(H_radio_fn(jnp.asarray(0.0)))
    heating_radio = jnp.full(n_stag, H_radio_per_cell)

    # Build component-isolated PhaseParams. Reuse the production
    # rheology/viscosity, only flip the transport flags.
    def with_flags(cond=False, conv=False, grav=False, mix=False, dil=False):
        return PhaseParams(
            phi_rheo=params0.phi_rheo,
            phi_width=params0.phi_width,
            viscosity_solid=10.0 ** float(params0.log10_visc_solid),
            viscosity_liquid=10.0 ** float(params0.log10_visc_liquid),
            grain_size=params0.grain_size,
            k_solid=params0.k_solid,
            k_liquid=params0.k_liquid,
            matprop_smooth_width=params0.matprop_smooth_width,
            conduction=cond, convection=conv,
            grav_sep=grav, mixing=mix, dilatation=dil,
            eddy_diff_thermal=params0.eddy_diff_thermal,
            eddy_diff_chemical=params0.eddy_diff_chemical,
            kappah_floor=params0.kappah_floor,
            bottom_up_grav_sep=True,
        )

    # Per-component isolation
    F_cond = np.asarray(compute_fluxes(
        S_stag, 0.0, eos, with_flags(cond=True), mesh, jnp.zeros(n_stag),
    ).heat_flux)
    F_conv = np.asarray(compute_fluxes(
        S_stag, 0.0, eos, with_flags(conv=True), mesh, jnp.zeros(n_stag),
    ).heat_flux)
    F_grav = np.asarray(compute_fluxes(
        S_stag, 0.0, eos, with_flags(grav=True), mesh, jnp.zeros(n_stag),
    ).heat_flux)
    F_mix = np.asarray(compute_fluxes(
        S_stag, 0.0, eos, with_flags(mix=True), mesh, jnp.zeros(n_stag),
    ).heat_flux)
    F_total = np.asarray(compute_fluxes(
        S_stag, 0.0, eos,
        with_flags(cond=True, conv=True, grav=True, mix=True),
        mesh, jnp.zeros(n_stag),
    ).heat_flux)

    # Sum of components
    F_sum = F_cond + F_conv + F_grav + F_mix
    component_residual = float(np.max(np.abs(F_total - F_sum)) /
                               max(np.max(np.abs(F_total)), 1e-30))
    print(f'component sum vs total: max rel diff = {component_residual:.3e}')

    # Internal heating (per-staggered-node) decomposition
    # Radio: just the constant H_radio_per_cell
    H_radio_arr = np.full(n_stag, H_radio_per_cell)
    # Dilatation: difference between heating_with_dil (all flags on,
    # dilatation ON) and heating_with_dil (all flags on, dilatation OFF).
    out_with_dil = compute_fluxes(
        S_stag, 0.0, eos,
        with_flags(cond=True, conv=True, grav=True, mix=True, dil=True),
        mesh, jnp.zeros(n_stag),
    )
    out_no_dil = compute_fluxes(
        S_stag, 0.0, eos,
        with_flags(cond=True, conv=True, grav=True, mix=True, dil=False),
        mesh, jnp.zeros(n_stag),
    )
    H_dil_arr = np.asarray(out_with_dil.heating - out_no_dil.heating)
    # Tidal: zero in this run (no tidal_array). Plot as zero baseline.
    H_tidal_arr = np.zeros(n_stag)
    H_total = H_radio_arr + H_dil_arr + H_tidal_arr

    # Save raw data
    np.savez(
        DATA / 'fig_02_flux_decomposition.npz',
        r_basic=r_basic, P_basic=P_basic,
        S_stag=np.asarray(S_stag),
        F_cond=F_cond, F_conv=F_conv, F_grav=F_grav, F_mix=F_mix,
        F_total=F_total, F_sum=F_sum,
        component_residual=component_residual,
        H_radio=H_radio_arr, H_dil=H_dil_arr, H_tidal=H_tidal_arr,
        H_total=H_total,
        n_stag=n_stag,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    # x-axis: depth from the surface (km). Surface at top, CMB at bottom.
    depth_km = (r_basic[-1] - r_basic) / 1000.0
    r_stag = np.asarray(mesh.radii_stag)
    depth_km_stag = (r_basic[-1] - r_stag) / 1000.0

    # Two panels: |F| (log-y, depth-x) and |H| (log-y, depth-x). Plot
    # absolute values with line style encoding sign so the comparison
    # of magnitudes is readable; the residual against the total is
    # printed in the caption.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True)

    def plot_signed_log(ax, x_km, y, color, label, lw=1.4, ls='-'):
        # Plot |y| with a marker change where y < 0; we draw a single
        # solid line and overlay open circles wherever y is negative
        # so signed information survives the log y-axis.
        absy = np.abs(y)
        ax.plot(x_km, np.maximum(absy, 1e-30),
                color=color, lw=lw, ls=ls, label=label)
        neg = y < 0
        if neg.any():
            ax.plot(x_km[neg], np.maximum(absy[neg], 1e-30),
                    color=color, lw=0, marker='v', markersize=3.5,
                    markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=0.7)

    # (a) Heat flux components
    ax = axes[0]
    plot_signed_log(ax, depth_km, F_cond, PALETTE['cond'],
                    r'$|F_\mathrm{cond}|$  (conduction)')
    plot_signed_log(ax, depth_km, F_conv, PALETTE['conv'],
                    r'$|F_\mathrm{conv}|$  (MLT convection)')
    plot_signed_log(ax, depth_km, F_grav, PALETTE['grav'],
                    r'$|F_\mathrm{grav}|$  (gravitational separation)')
    plot_signed_log(ax, depth_km, F_mix, PALETTE['mix'],
                    r'$|F_\mathrm{mix}|$  (chemical mixing)')
    plot_signed_log(ax, depth_km, F_total, PALETTE['total'],
                    r'$|F_\mathrm{tot}|$  = sum',
                    lw=2.0, ls='--')
    ax.set_yscale('log')
    ax.set_ylabel(r'$|F|$  (W m$^{-2}$)')
    ax.set_ylim(1e-2, 1e13)
    ax.set_xlim(0, depth_km[0])
    ax.legend(loc='center right', fontsize=7.5, ncol=1, framealpha=0.92)
    ax.grid(alpha=0.3, which='both')
    ax.set_title(r'Heat flux decomposition  (mushy state, $\phi\approx 0.5$).'
                 r'  Open triangles mark cells where the signed flux is negative.')
    panel_label(ax, '(a)', loc='lower left')

    # (b) Internal heating sources
    ax = axes[1]
    plot_signed_log(ax, depth_km_stag, H_radio_arr, PALETTE['radio'],
                    r'$|H_\mathrm{radio}|$  (4-isotope cocktail at $t=0$)')
    plot_signed_log(ax, depth_km_stag, H_dil_arr, PALETTE['dil'],
                    r'$|H_\mathrm{dil}|$  (PdV term)')
    # Tidal is identically zero in this config; show as floored line for legend.
    ax.plot(depth_km_stag, np.full(n_stag, 1e-20), color=PALETTE['tidal'],
            lw=1.0, ls=':', label=r'$H_\mathrm{tidal}$ (= 0 in this run)')
    plot_signed_log(ax, depth_km_stag, H_total, PALETTE['total'],
                    r'$|H_\mathrm{tot}|$  = sum',
                    lw=2.0, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('Depth from surface (km)')
    ax.set_ylabel(r'$|H|$  (W kg$^{-1}$)')
    ax.set_ylim(1e-19, 1e4)
    ax.set_xlim(0, depth_km_stag[0])
    ax.legend(loc='center right', fontsize=7.5, framealpha=0.92)
    ax.grid(alpha=0.3, which='both')
    ax.set_title('Internal volumetric heat sources at staggered nodes')
    panel_label(ax, '(b)', loc='lower left')

    fig.tight_layout()
    save(fig, OUT / 'fig_02_flux_decomposition.pdf')

    print('fig_02 saved.')
    print(f'  H_radio per cell = {H_radio_per_cell:.3e} W/kg')
    print(f'  max |H_dil|      = {np.abs(H_dil_arr).max():.3e} W/kg')
    print(f'  max |F_total|    = {np.abs(F_total).max():.3e} W/m^2')


if __name__ == '__main__':
    main()
