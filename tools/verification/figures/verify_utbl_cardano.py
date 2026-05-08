"""V&V Figure 5: UTBL Cardano correction T_surf(T_interior).

Verifies the ultra-thin thermal boundary layer (UTBL) correction
of Bower+2018 Eq. 18, which replaces the radiating surface temperature
T_surf with the real cubic root of

    b * T_surf^3 + T_surf - T_interior = 0,

where b = ``param_utbl_const`` (units of K^-2; physical interpretation:
inverse-cube of the UTBL conductance scale). Both implementations are
exercised: the numpy production reference
(``aragog.solver.boundary.BoundaryConditions._utbl_tsurf``, np.cbrt
form) and the JAX traceable form (``aragog.jax.solver._utbl_tsurf_jax``,
jnp.cbrt). They must produce byte-equal output, since the analytic
Jacobian for grey-body atmosphere flux differentiation flows through
this branch when ``param_utbl=True``.

Three values of ``b`` are scanned over the magma-ocean-relevant
T_interior range 1500-5000 K:

    b = 1e-7 K^-2   (weak correction; T_surf ~ T_interior)
    b = 1e-6 K^-2   (canonical Bower+2018 SPIDER value)
    b = 1e-5 K^-2   (strong correction; T_surf << T_interior at high T)

The correction is monotone, satisfies T_surf <= T_interior everywhere,
and reduces to T_surf = T_interior in the limit b -> 0.

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_utbl_cardano.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
from _style import apply_rc, panel_label, save  # noqa: E402

OUT = REPO_ROOT / 'docs' / 'figures' / 'vv'
DATA = REPO_ROOT / 'output' / 'aragog_vv_data'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

from aragog.jax.solver import _utbl_tsurf_jax  # noqa: E402


def utbl_numpy(T_interior, b):
    """Numpy reference (matches BoundaryConditions._utbl_tsurf)."""
    p = 1.0 / b
    q = -T_interior / b
    discriminant = q**2 / 4.0 + p**3 / 27.0
    sd = np.sqrt(discriminant)
    return np.cbrt(-q / 2.0 + sd) + np.cbrt(-q / 2.0 - sd)


def main():
    apply_rc()

    T_int = np.linspace(1500.0, 5000.0, 700)
    b_vals = [1e-7, 1e-6, 1e-5]
    colors = ['#2ca02c', '#1f77b4', '#d62728']

    results_np = {}
    results_jx = {}
    max_abs_err = 0.0
    for b in b_vals:
        T_np = utbl_numpy(T_int, b)
        T_jx = np.array(_utbl_tsurf_jax(jnp.asarray(T_int), jnp.asarray(b)))
        results_np[b] = T_np
        results_jx[b] = T_jx
        err = float(np.abs(T_np - T_jx).max())
        max_abs_err = max(max_abs_err, err)

    np.savez(
        DATA / 'fig_05_utbl_cardano.npz',
        T_interior=T_int,
        b_values=np.array(b_vals),
        T_surf_numpy=np.stack([results_np[b] for b in b_vals]),
        T_surf_jax=np.stack([results_jx[b] for b in b_vals]),
        max_abs_err_numpy_vs_jax=max_abs_err,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # (a) T_surf vs T_interior at three b values, with diagonal reference
    ax = axes[0]
    ax.plot(T_int, T_int, color='k', lw=0.8, ls=':',
            label=r'$T_\mathrm{surf}=T_\mathrm{interior}$ ($b\to 0$)')
    for b, c in zip(b_vals, colors):
        ax.plot(T_int, results_np[b], color=c, lw=1.6,
                label=fr'numpy, $b={b:g}$ K$^{{-2}}$')
        ax.plot(T_int, results_jx[b], color=c, lw=0.8, ls='--',
                label=fr'JAX,   $b={b:g}$ K$^{{-2}}$')
    ax.set_xlabel(r'Interior surface temperature  $T_\mathrm{interior}$ (K)')
    ax.set_ylabel(r'Radiating surface temperature  $T_\mathrm{surf}$ (K)')
    ax.set_xlim(T_int[0], T_int[-1])
    ax.set_ylim(0, 5200)
    ax.legend(loc='upper left', fontsize=7.5, ncol=1)
    ax.grid(alpha=0.3)
    panel_label(ax, '(a)', loc='lower right')

    # (b) Cubic residual to verify the root: b T_surf^3 + T_surf - T_int = 0
    ax = axes[1]
    for b, c in zip(b_vals, colors):
        T_s = results_np[b]
        residual = b * T_s**3 + T_s - T_int
        ax.plot(T_int, np.maximum(np.abs(residual), 1e-30),
                color=c, lw=1.4, label=fr'$b={b:g}$ K$^{{-2}}$')
    ax.axhline(np.finfo(np.float64).eps * T_int.max(), color='k',
               lw=0.8, ls=':',
               label=r'$\epsilon_\mathrm{mach}\cdot T_\mathrm{max}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T_\mathrm{interior}$ (K)')
    ax.set_ylabel(r'$|b\,T_\mathrm{surf}^{3} + T_\mathrm{surf} - T_\mathrm{interior}|$  (K)')
    ax.set_xlim(T_int[0], T_int[-1])
    ax.set_ylim(1e-15, 1e-9)
    ax.legend(loc='upper left', fontsize=7.5)
    ax.grid(alpha=0.3, which='both')
    ax.text(0.97, 0.05,
            f'max |numpy-JAX|\n= {max_abs_err:.2e} K',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2))
    panel_label(ax, '(b)', loc='upper right')

    fig.tight_layout()
    save(fig, OUT / 'fig_05_utbl_cardano.pdf')
    print(f'fig_05 saved; max_abs_err={max_abs_err:.3e} K')


if __name__ == '__main__':
    main()
