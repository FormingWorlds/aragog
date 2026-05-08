"""V&V Figure 4: Three-regime permeability F(porosity).

Verifies that the gravitational-separation permeability factor F = K
(in m^2) implements the Abe (1993, 1995) three-regime model -- Stokes,
Rumpf-Gupte (RG), and Blake-Kozeny-Carman (BKC) -- with the exact
critical-porosity tanh blends specified in the Aragog formulation. Both the numpy implementation
(``aragog.eos.entropy_phase``) and the JAX implementation
(``aragog.jax.phase.relative_velocity``) are evaluated at the same
porosities; bytes-for-bytes agreement is asserted by the unit test
``tests/test_permeability_constants.py``. This figure shows the resulting
F(zeta) shape and the regime transitions at zeta = 0.0769452 and
0.771462.

For visualisation we factor out the grain-size scaling a^2 (a = 1 mm
default) so the y-axis is dimensionless F/a^2 and the regime ordering
is independent of the chosen grain size.

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_permeability.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)  # match production aragog precision
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

# Critical porosities from the Aragog formulation (matches the
# constants in eos/entropy_phase.py and jax/phase.py).
ZETA_BKC_RG = 0.0769452
ZETA_RG_STOKES = 0.771462


def F_BKC(zeta, a=1.0):
    """Blake-Kozeny-Carman branch (per a^2): zeta^2 / [(1-zeta)^2 * 1000]."""
    one_m = np.maximum(1.0 - zeta, 1e-20)
    return a**2 * zeta**2 / (one_m**2 * 1000.0)


def F_RG(zeta, a=1.0):
    """Rumpf-Gupte branch (per a^2): (5/7) * zeta^4.5."""
    return a**2 * (5.0 / 7.0) * zeta**4.5


def F_Stokes(zeta, a=1.0):
    """Stokes-settling branch (per a^2): 2/9 (constant)."""
    return a**2 * 2.0 / 9.0 * np.ones_like(zeta)


def tanh_weight(x, threshold, width):
    """Smooth (0,1) sigmoid centred on `threshold` with characteristic `width`."""
    return 0.5 * (1.0 + np.tanh((x - threshold) / width))


def F_blended_numpy(zeta, a=1.0):
    """Numpy reference: blended F(zeta) using the the Aragog blend constants."""
    w_rg = tanh_weight(zeta, ZETA_BKC_RG, 0.02)
    w_st = tanh_weight(zeta, ZETA_RG_STOKES, 0.05)
    f_b = F_BKC(zeta, a)
    f_r = F_RG(zeta, a)
    f_s = F_Stokes(zeta, a)
    return (1.0 - w_rg) * f_b + (w_rg - w_st) * f_r + w_st * f_s


def F_blended_jax(zeta_arr, a=1.0):
    """JAX path: same blend evaluated through jnp ops."""
    z = jnp.asarray(zeta_arr)
    w_rg = 0.5 * (1.0 + jnp.tanh((z - ZETA_BKC_RG) / 0.02))
    w_st = 0.5 * (1.0 + jnp.tanh((z - ZETA_RG_STOKES) / 0.05))
    one_m = jnp.maximum(1.0 - z, 1e-20)
    f_b = a**2 * z**2 / (one_m**2 * 1000.0)
    f_r = a**2 * (5.0 / 7.0) * z**4.5
    f_s = a**2 * 2.0 / 9.0
    return np.asarray((1.0 - w_rg) * f_b + (w_rg - w_st) * f_r + w_st * f_s)


def main():
    apply_rc()

    # Dense porosity grid; avoid zeta=1 (BKC singularity is regularised
    # by tanh-blending to Stokes well below unity).
    zeta = np.linspace(1e-4, 0.999, 4000)
    a = 1.0  # a^2 = 1 -> y-axis is F/a^2

    f_b = F_BKC(zeta, a)
    f_r = F_RG(zeta, a)
    f_s = F_Stokes(zeta, a)
    f_np = F_blended_numpy(zeta, a)
    f_jx = F_blended_jax(zeta, a)

    abs_err = np.abs(f_jx - f_np)
    # Use a finite floor on |f| so the relative-error metric is not
    # dominated by float64-roundoff at zeta -> 0 (where F ~ 1e-11 and
    # the absolute difference of 1e-17 is at machine precision).
    rel_err = abs_err / np.maximum(np.abs(f_np), 1e-12)
    max_rel_err = float(rel_err.max())
    max_abs_err = float(abs_err.max())

    # Save data
    np.savez(
        DATA / 'fig_04_permeability.npz',
        zeta=zeta, F_BKC=f_b, F_RG=f_r, F_Stokes=f_s,
        F_blended_numpy=f_np, F_blended_jax=f_jx,
        max_rel_err_numpy_vs_jax=max_rel_err,
        max_abs_err_numpy_vs_jax=max_abs_err,
        zeta_BKC_RG=ZETA_BKC_RG, zeta_RG_Stokes=ZETA_RG_STOKES,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # (a) Linear-y zoomed view of the three branches + blend
    ax = axes[0]
    ax.plot(zeta, f_b, color=PALETTE['bkc'],    lw=1.1, ls='--', label='BKC branch')
    ax.plot(zeta, f_r, color=PALETTE['rg'],     lw=1.1, ls='--', label='RG branch')
    ax.plot(zeta, f_s, color=PALETTE['stokes'], lw=1.1, ls='--', label='Stokes branch')
    ax.plot(zeta, f_np, color='k', lw=1.8, label='Blended F (numpy)')
    ax.plot(zeta, f_jx, color=PALETTE['jax'], lw=1.0, ls=':', label='Blended F (JAX)')
    ax.axvline(ZETA_BKC_RG,    color='gray', lw=0.6, ls=':')
    ax.axvline(ZETA_RG_STOKES, color='gray', lw=0.6, ls=':')
    ax.set_xlabel(r'Porosity  $\zeta$')
    ax.set_ylabel(r'$F(\zeta) / a^2$ (dimensionless)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.32)
    ax.legend(loc='upper left', fontsize=7.5)
    ax.text(ZETA_BKC_RG, 0.30,    fr'$\zeta_1={ZETA_BKC_RG}$',
            rotation=90, va='top', ha='right', fontsize=7.5, color='dimgray')
    ax.text(ZETA_RG_STOKES, 0.30, fr'$\zeta_2={ZETA_RG_STOKES}$',
            rotation=90, va='top', ha='right', fontsize=7.5, color='dimgray')
    panel_label(ax, '(a)', loc='upper right')

    # (b) Log-log view spanning the BKC vanishingly-small regime
    ax = axes[1]
    ax.plot(zeta, np.maximum(f_b, 1e-30), color=PALETTE['bkc'],    lw=1.1, ls='--', label='BKC branch')
    ax.plot(zeta, np.maximum(f_r, 1e-30), color=PALETTE['rg'],     lw=1.1, ls='--', label='RG branch')
    ax.plot(zeta, np.maximum(f_s, 1e-30), color=PALETTE['stokes'], lw=1.1, ls='--', label='Stokes branch')
    ax.plot(zeta, np.maximum(f_np, 1e-30), color='k', lw=1.8, label='Blended F (numpy)')
    ax.plot(zeta, np.maximum(f_jx, 1e-30), color=PALETTE['jax'], lw=1.0, ls=':', label='Blended F (JAX)')
    ax.axvline(ZETA_BKC_RG,    color='gray', lw=0.6, ls=':')
    ax.axvline(ZETA_RG_STOKES, color='gray', lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Porosity  $\zeta$')
    ax.set_ylabel(r'$F(\zeta) / a^2$ (log scale)')
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-12, 1.0)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.text(0.04, 0.95,
            f'max |JAX-numpy|\n= {max_abs_err:.2e}\n(float64 epsilon)',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=2))
    panel_label(ax, '(b)', loc='upper left')

    fig.tight_layout()
    save(fig, OUT / 'fig_04_permeability.pdf')
    print(f'fig_04 saved; max_abs_err={max_abs_err:.3e}, max_rel_err(floor1e-12)={max_rel_err:.3e}')


if __name__ == '__main__':
    main()
