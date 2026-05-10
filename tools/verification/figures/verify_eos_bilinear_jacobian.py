"""V&V Figure 3: Bilinear EOS Jacobian within-cell exactness.

Verifies that the JAX bilinear EOS interpolator
``aragog.jax.eos._bilinear_interp`` returns the EXACT analytical
gradient of the bilinear function within each (P, S) grid cell.
The integer searchsorted indices that look up the corner values are
wrapped in ``jax.lax.stop_gradient`` so the autodiff path sees only
the smooth bilinear closed-form

    f(tp, ts) = f00*(1-tp)*(1-ts) + f10*tp*(1-ts)
              + f01*(1-tp)*ts     + f11*tp*ts,

with tp = (P - P_i)/dP_i and ts = (S - S_j)/dS_j on cell [i,i+1] x
[j,j+1]. ``df/dP`` and ``df/dS`` are then constants WITHIN each cell
(the analytical bilinear gradient at the lower-left corner) and jump
discontinuously at the cell boundaries when the underlying values
field is non-linear (= the documented and audited behaviour).

The figure shows two panels:

  (a) ``df/dP`` along a constant-S slice for a smooth pure-bilinear
      values field (linear in P, S). JAX-jacrev and the analytical
      gradient agree at machine precision; the gradient is constant
      across all grid cells.

  (b) Same slice for a non-linear values field where the slope jumps
      between adjacent cells. JAX-jacrev follows the local cell's
      bilinear gradient exactly and shows the documented jumps at
      the cell boundaries (CVODE absorbs these as stiff-region step
      rejections; see ``aragog/jax/eos.py``).

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_eos_bilinear_jacobian.py
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
from _style import PALETTE, apply_rc, panel_label, save  # noqa: E402

OUT = REPO_ROOT / 'docs' / 'figures' / 'vv'
DATA = REPO_ROOT / 'output' / 'aragog_vv_data'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

from aragog.jax.eos import _bilinear_interp  # noqa: E402


def jax_jacobian_dP(P_grid, S_grid, vals_arr, P_query, S_query):
    """Per-query-point ∂f/∂P via jax.grad."""
    def f_of_P(p):
        return _bilinear_interp(
            P_grid, S_grid, vals_arr,
            jnp.atleast_1d(p), jnp.atleast_1d(S_query),
        )[0]
    g = jax.grad(f_of_P)
    return float(g(jnp.asarray(P_query)))


def analytical_dP(P_grid, S_grid, vals_arr, P_query, S_query):
    """Analytical bilinear ∂f/∂P at (P_query, S_query)."""
    P = np.asarray(P_grid)
    S = np.asarray(S_grid)
    v = np.asarray(vals_arr)
    # locate cell (clamped inside the table)
    i = int(np.clip(np.searchsorted(P, P_query) - 1, 0, len(P) - 2))
    j = int(np.clip(np.searchsorted(S, S_query) - 1, 0, len(S) - 2))
    dP = P[i + 1] - P[i]
    dS = S[j + 1] - S[j]
    ts = (S_query - S[j]) / dS
    # ∂f/∂P inside this cell:
    #   df/dP = (1/dP) [(v[i+1,j]-v[i,j])*(1-ts) + (v[i+1,j+1]-v[i,j+1])*ts]
    return ((v[i + 1, j] - v[i, j]) * (1.0 - ts)
            + (v[i + 1, j + 1] - v[i, j + 1]) * ts) / dP


def main():
    apply_rc()

    # ── (a) Pure-bilinear field on a 6x6 grid ─────────────────────────
    # values[i,j] = 100 i + j  ->  uniform ∂f/∂P = 100/dP, ∂f/∂S = 1/dS
    # everywhere, so JAX should reproduce the constant 100 across all
    # cells with no jumps.
    P_a = jnp.linspace(1.0, 6.0, 6)
    S_a = jnp.linspace(10.0, 60.0, 6)
    vals_a = jnp.array([[100.0 * i + j for j in range(6)] for i in range(6)])

    P_query_a = np.linspace(1.05, 5.95, 400)
    S_slice_a = 35.0  # mid-grid in S
    df_jax_a = np.array([
        jax_jacobian_dP(P_a, S_a, vals_a, p, S_slice_a) for p in P_query_a
    ])
    df_an_a = np.array([
        analytical_dP(P_a, S_a, vals_a, p, S_slice_a) for p in P_query_a
    ])
    err_a = np.abs(df_jax_a - df_an_a)
    max_err_a = float(err_a.max())

    # ── (b) Non-linear field ──────────────────────────────────────────
    # Make ∂f/∂P jump between successive cells by hand:
    # row P_i: v(S_j) = (i**2 + j) * 10
    # ∂f/∂P|cell[i,i+1] = (((i+1)^2 - i^2) * 10) / dP = 10 * (2 i + 1) / dP
    P_b = jnp.linspace(1.0, 6.0, 6)
    S_b = jnp.linspace(10.0, 60.0, 6)
    vals_b = jnp.array([
        [10.0 * (i**2 + j) for j in range(6)] for i in range(6)
    ])

    P_query_b = np.linspace(1.05, 5.95, 800)
    S_slice_b = 35.0
    df_jax_b = np.array([
        jax_jacobian_dP(P_b, S_b, vals_b, p, S_slice_b) for p in P_query_b
    ])
    df_an_b = np.array([
        analytical_dP(P_b, S_b, vals_b, p, S_slice_b) for p in P_query_b
    ])
    err_b = np.abs(df_jax_b - df_an_b)
    max_err_b = float(err_b.max())

    # ── Save raw data ─────────────────────────────────────────────────
    np.savez(
        DATA / 'fig_03_eos_bilinear_jacobian.npz',
        P_query_a=P_query_a, df_jax_a=df_jax_a, df_analytic_a=df_an_a,
        P_query_b=P_query_b, df_jax_b=df_jax_b, df_analytic_b=df_an_b,
        max_err_smooth_field=max_err_a,
        max_err_nonlinear_field=max_err_b,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # (a) Smooth bilinear field
    ax = axes[0]
    ax.plot(P_query_a, df_an_a,
            color=PALETTE['analytic'], lw=2.0,
            label='Analytical bilinear gradient')
    ax.plot(P_query_a, df_jax_a,
            color=PALETTE['jax'], lw=1.0, ls='--',
            label='JAX jax.grad')
    for p_grid in np.asarray(P_a)[1:-1]:
        ax.axvline(p_grid, color='gray', lw=0.5, ls=':')
    ax.set_xlabel(r'Pressure  $P$  (table units)')
    ax.set_ylabel(r'$\partial f / \partial P$  (table units)')
    ax.set_xlim(P_query_a[0], P_query_a[-1])
    ax.set_title('Linear field: gradient is constant')
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.04, 0.95,
            f'max |JAX - analytic|\n= {max_err_a:.2e}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2))
    panel_label(ax, '(a)', loc='upper right')

    # (b) Non-linear field with cell-boundary jumps
    ax = axes[1]
    ax.step(P_query_b, df_an_b, where='mid',
            color=PALETTE['analytic'], lw=2.0,
            label='Analytical bilinear gradient')
    ax.plot(P_query_b, df_jax_b,
            color=PALETTE['jax'], lw=1.0, ls='--',
            label='JAX jax.grad')
    for p_grid in np.asarray(P_b)[1:-1]:
        ax.axvline(p_grid, color='gray', lw=0.5, ls=':')
    ax.set_xlabel(r'Pressure  $P$  (table units)')
    ax.set_ylabel(r'$\partial f / \partial P$  (table units)')
    ax.set_xlim(P_query_b[0], P_query_b[-1])
    ax.set_title('Non-linear field: jumps at cell boundaries')
    ax.legend(loc='upper left', fontsize=8)
    ax.text(0.04, 0.62,
            f'max |JAX - analytic|\n(within-cell)\n= {max_err_b:.2e}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2))
    panel_label(ax, '(b)', loc='upper right')

    fig.tight_layout()
    save(fig, OUT / 'fig_03_eos_bilinear_jacobian.pdf')
    print(f'fig_03 saved; smooth-field max err={max_err_a:.3e}, '
          f'non-linear within-cell max err={max_err_b:.3e}')


if __name__ == '__main__':
    main()
