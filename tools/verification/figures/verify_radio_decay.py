"""V&V Figure 6: Radiogenic decay curves over the Solar System age.

Verifies that ``aragog.jax.solver.make_radio_heating_fn`` reproduces the
analytical exponential decay law

    H_radio,i(t) = phi_i * chi_i * h_i * exp(-ln(2) * (t - t0_i) / tau_{1/2,i})

per Aragog's radiogenic source term and the
parameters tabulated in ``aragog/cfg/abe_mixed_init.cfg`` (Ruedas 2017,
Turcotte & Schubert 2014). The analytical form is evaluated independently
in numpy and the JAX path is queried point-wise; the two curves should
overlap at machine precision over the full 0 -> 4.55 Gyr range.

Run:
    conda activate proteus  # or your aragog env
    python tools/verification/figures/verify_radio_decay.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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

from aragog.jax.solver import make_radio_heating_fn  # noqa: E402

# Parameters from abe_mixed_init.cfg (Ruedas 2017; Turcotte & Schubert
# 2014). Concentrations are ppm of the parent element; abundance is the
# isotopic fraction; heat_production is W/kg of pure isotope; t0 is the
# reference time at which abundance/concentration are quoted.
#
# K40, Th232, U235, U238 are the four long-lived nuclides quoted at the
# present-day Earth abundance with t0 = 4.55 Gyr (Solar System
# formation, our t = 0 on the plot). Aragog's analytical decay law then
# amplifies them by 2^(t0/half_life) at t = 0 to recover the
# Solar-System-formation abundance.
#
# Al26 and Fe60 are short-lived nuclides that contribute only during
# the first few Myr after formation. They are extinct today, so the
# bundled cfg quotes them at zero present-day abundance; the values
# below pin them at their canonical Solar System initial (SSI)
# abundances at t0 = 0 (formation), with parent-element concentrations
# from the bulk silicate Earth composition of McDonough & Sun (1995):
#
#   - (26Al/27Al)_0 = 5.25e-5   Jacobsen et al. (2008) canonical CAI value;
#                               same value used by Lichtenberg et al. (2016, 2021).
#   - (60Fe/56Fe)_0 = 1.0e-8    Tang & Dauphas (2012) consensus SSI value
#                               (older estimates ~1e-6 are now disfavoured).
#   - [Al]_BSE = 2.36 wt% = 23600 ppm
#   - [Fe]_BSE = 6.26 wt% = 62600 ppm
#
# Showing them on the plot makes the dominant Al-26 short-lived
# contribution at t = 0 visible alongside the four long-lived isotopes;
# the curves for Al-26 and Fe-60 then drop by 60+ orders of magnitude
# within a few tens of Myr, leaving the long-lived nuclides to carry
# the heat budget for the rest of the Solar System lifetime.
ISOTOPES = {
    'K40':   dict(t0=4.55e9, ab=1.1668e-4, conc_ppm=310.0,    hp=2.8761e-5, hl=1.248e9, color=PALETTE['K40']),
    'Th232': dict(t0=4.55e9, ab=1.0,        conc_ppm=0.124,    hp=2.6368e-5, hl=14.0e9,  color=PALETTE['Th232']),
    'U235':  dict(t0=4.55e9, ab=7.2045e-3, conc_ppm=0.031,    hp=5.68402e-4, hl=0.704e9, color=PALETTE['U235']),
    'U238':  dict(t0=4.55e9, ab=0.9927955, conc_ppm=0.031,    hp=9.4946e-5, hl=4.468e9, color=PALETTE['U238']),
    'Al26':  dict(t0=0.0,    ab=5.25e-5,    conc_ppm=23600.0,  hp=0.3583,    hl=7.17e5,  color=PALETTE.get('Al26', '#d62728')),
    'Fe60':  dict(t0=0.0,    ab=1.0e-8,     conc_ppm=62600.0,  hp=3.6579e-2, hl=2.62e6,  color=PALETTE.get('Fe60', '#9467bd')),
}

LOG2 = np.log(2.0)


def H_iso_analytical(t_yr, p):
    """Analytical per-isotope heating rate [W/kg of bulk silicate]."""
    conc = p['conc_ppm'] * 1.0e-6  # ppm -> mass fraction of parent
    return p['hp'] * p['ab'] * conc * np.exp(LOG2 * (p['t0'] - t_yr) / p['hl'])


def main():
    apply_rc()

    # ── Build the JAX radio function with all four isotopes (matches
    # the per-cell uniform sum in compute_fluxes) ─────────────────────
    hp = np.array([ISOTOPES[k]['hp'] for k in ISOTOPES])
    ab = np.array([ISOTOPES[k]['ab'] for k in ISOTOPES])
    cc = np.array([ISOTOPES[k]['conc_ppm'] * 1.0e-6 for k in ISOTOPES])
    t0 = np.array([ISOTOPES[k]['t0'] for k in ISOTOPES])
    hl = np.array([ISOTOPES[k]['hl'] for k in ISOTOPES])
    H_jax = make_radio_heating_fn(hp, ab, cc, t0, hl)

    # Time axis: log-spaced over the full Solar-System lifetime to keep
    # the short-lived (Al-26, Fe-60) era visible alongside the
    # long-lived (K, Th, U) era. A linear axis would clip the short-
    # lived decays into the first sample and hide them.
    t_yr = np.geomspace(1.0e3, 4.55e9, 1200)

    # Per-isotope (numpy analytical)
    H_per = {k: H_iso_analytical(t_yr, p) for k, p in ISOTOPES.items()}
    H_total_np = np.sum(np.stack(list(H_per.values()), axis=0), axis=0)

    # JAX total (vectorised over t)
    H_total_jax = np.array(jax.vmap(H_jax)(jnp.asarray(t_yr)))

    # Parity check ────────────────────────────────────────────────────
    rel_err = np.abs(H_total_jax - H_total_np) / np.maximum(H_total_np, 1e-30)
    max_rel_err = float(rel_err.max())

    # ── Save raw data for reproducibility ─────────────────────────────
    np.savez(
        DATA / 'fig_06_radio_decay.npz',
        t_yr=t_yr,
        H_K40=H_per['K40'], H_Th232=H_per['Th232'],
        H_U235=H_per['U235'], H_U238=H_per['U238'],
        H_Al26=H_per['Al26'], H_Fe60=H_per['Fe60'],
        H_total_numpy=H_total_np, H_total_jax=H_total_jax,
        max_rel_err_numpy_vs_jax=max_rel_err,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)

    # (a) Per-isotope + total, log y
    ax = axes[0]
    t_myr = t_yr / 1.0e6
    for name, p in ISOTOPES.items():
        ax.plot(t_myr, H_per[name],
                color=p['color'], lw=1.4, label=name)
    ax.plot(t_myr, H_total_np,
            color='k', lw=2.0, ls='-', label='Total (numpy)')
    ax.plot(t_myr, H_total_jax,
            color=PALETTE['jax'], lw=1.0, ls='--', label='Total (JAX)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time since formation (Myr)')
    ax.set_ylabel(r'$H_\mathrm{radio}$  (W kg$^{-1}$)')
    ax.set_xlim(t_myr[0], t_myr[-1])
    # Y range covers the Al-26-dominated SSI peak at t -> 0 (~1e-6 W/kg)
    # down to a few orders below the present-day total (~1e-12 W/kg).
    ax.set_ylim(1e-15, 5e-6)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower left', ncol=2, frameon=True, fontsize=7.5)
    panel_label(ax, '(a)', loc='upper right')

    # (b) Numpy-vs-JAX relative error
    ax = axes[1]
    ax.plot(t_myr, np.maximum(rel_err, 1e-18),
            color=PALETTE['jax'], lw=1.2)
    ax.axhline(np.finfo(np.float64).eps, color='k', lw=0.8, ls=':',
               label=r'$\epsilon_\mathrm{mach}$ (float64)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time since formation (Myr)')
    ax.set_ylabel('|JAX - numpy| / numpy')
    ax.set_xlim(t_myr[0], t_myr[-1])
    ax.set_ylim(1e-18, 1e-12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.text(0.05, 0.95,
            f'max rel. err.\n= {max_rel_err:.2e}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=2))
    panel_label(ax, '(b)', loc='upper left')

    fig.tight_layout()
    save(fig, OUT / 'fig_06_radio_decay.pdf')
    print(f'fig_06 saved; max_rel_err={max_rel_err:.3e}')


if __name__ == '__main__':
    main()
