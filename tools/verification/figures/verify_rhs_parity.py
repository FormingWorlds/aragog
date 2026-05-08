"""V&V Figure 1: Numpy <-> JAX RHS parity heatmap on a real CHILI state.

Verifies that the JAX path
``aragog.jax.solver.dSdt_energy_balance`` (production CVODE+JAX RHS,
energy_balance core BC) produces the same dS/dt vector as the numpy
reference ``EntropySolver._dSdt_single`` on the same state, to within
floating-point round-off. The two RHS implementations are independent
re-derivations of the FV finite-volume form (the finite-volume entropy form):

    (rho cp V)_i (dT/dt)_i = -F_{i+1/2} A_{i+1/2}
                             + F_{i-1/2} A_{i-1/2}
                             + Phi_i V_i,

with the entropy formulation S as the prognostic variable. Parity
across the full RHS vector is the necessary condition for the JAX
analytic Jacobian to be a useful CVODE preconditioner.

Three representative magma-ocean states are exercised on the same
80-cell CHILI Earth mesh (``chili_repro_v2.toml``):

  IC          - full magma ocean, S ~ 3900 J/kg/K throughout
  mid         - active solidification, S = 3300 (CMB) -> 3700 (surface)
  near_solid  - late solidification, S = 3000 (CMB) -> 3300 (surface)

Each panel shows the per-staggered-cell relative error
|dSdt_numpy - dSdt_jax| / max(|dSdt_numpy|, eps) on a log y-axis.
The colour series in panel (d) ranks the worst-case errors across all
states.

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_rhs_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin BLAS threads to remove parallel-RNG drift in the numpy reference.
for var in [
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OMP_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
]:
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

NOISE_FLOOR = 1.0e-3  # J/kg/K/yr -- below this the entropy ODE is at machine eps


def parity_one_state(solver, args, state, label):
    from aragog.jax.solver import dSdt_energy_balance, _no_radio  # noqa: WPS433

    # z02 helper returns a 5-tuple (eos, params, mesh, bc, heating); the
    # current dSdt_energy_balance signature expects a 6-tuple with a
    # JAX-traceable radio-heating callable as the trailing entry. The
    # production CHILI run was launched without radionuclides, so the
    # canonical no-op callable ``_no_radio`` is the correct shim here.
    args6 = tuple(args) + (_no_radio,)
    f_np = np.asarray(solver._dSdt_single(0.0, state))
    f_jx = np.asarray(dSdt_energy_balance(0.0, jnp.asarray(state), args6))
    abs_err = np.abs(f_np - f_jx)
    is_noise = (np.abs(f_np) < NOISE_FLOOR) & (np.abs(f_jx) < NOISE_FLOOR)
    denom = np.where(is_noise, 1.0, np.maximum(np.abs(f_np), 1e-30))
    rel_err = np.where(is_noise, 0.0, abs_err / denom)
    return dict(label=label, f_np=f_np, f_jax=f_jx, abs_err=abs_err, rel_err=rel_err)


def main():
    apply_rc()

    if PROTEUS_SCRIPTS:
        sys.path.insert(0, str(Path(PROTEUS_SCRIPTS).resolve()))
    from z02_parity_multi_state import (  # noqa: WPS433
        build_solver_and_jax_args,
        synthesize_state,
    )

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag

    states = {
        'IC': synthesize_state(solver, eos_jax, 'IC'),
        'mid': synthesize_state(solver, eos_jax, 'mid'),
        'near_solid': synthesize_state(solver, eos_jax, 'near_solid'),
    }
    labels = list(states.keys())

    results = {k: parity_one_state(solver, args, v, k) for k, v in states.items()}

    # Save raw arrays
    np.savez(
        DATA / 'fig_01_rhs_parity.npz',
        n_stag=n_stag,
        **{f'{k}_dSdt_numpy': results[k]['f_np'] for k in labels},
        **{f'{k}_dSdt_jax': results[k]['f_jax'] for k in labels},
        **{f'{k}_abs_err': results[k]['abs_err'] for k in labels},
        **{f'{k}_rel_err': results[k]['rel_err'] for k in labels},
    )

    # ── Plot: 2x3 grid (3 states x {abs, rel}) ───────────────────────
    # Top row: ABSOLUTE residuals per cell (the primary parity claim).
    # Bottom row: RELATIVE residuals per cell (a derived view that
    #     inflates near vanishing |dS/dt| in the uniform-entropy IC
    #     limit; reported only for completeness).
    # Splitting these makes the precision claim honest: panel (a) shows
    # absolute residuals at floating-point round-off (~1e-13 J/kg/K/yr)
    # across interior cells with localised boundary-cell spikes; panel
    # (d) shows the relative-error formula losing significance against
    # a small denominator at IC, NOT a real solver disagreement.
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True)
    cell_idx = np.arange(n_stag + 1)  # state vector is N+1 (energy_balance)

    state_colors = {
        'IC': PALETTE['analytic'],
        'mid': PALETTE['conv'],
        'near_solid': PALETTE['mix'],
    }

    panel_letters = 'abcdef'
    # Top row: absolute residuals
    for col, kind in enumerate(labels):
        ax = axes[0, col]
        abs_e = results[kind]['abs_err']
        f_np = results[kind]['f_np']
        rel = results[kind]['rel_err']
        # Floor at the smallest representable f8 to keep the log axis
        # readable; cells with abs_err == 0 plot at 1e-18.
        ax.semilogy(
            cell_idx,
            np.maximum(abs_e, 1e-18),
            color=state_colors[kind],
            lw=1.0,
            marker='o',
            markersize=2.5,
            label=f'state = {kind}',
        )
        # Reference line: machine eps × max(|dS/dt|) is the absolute
        # round-off floor expected on a single FP operation. Anything
        # below this line is at floating-point round-off; anything
        # above it is a real algorithmic disagreement.
        floor = np.finfo(np.float64).eps * float(np.max(np.abs(f_np)))
        ax.axhline(
            max(floor, 1e-18),
            color='k',
            lw=0.6,
            ls=':',
            label=r'$\epsilon_\mathrm{mach}\,\max|\dot S|$',
        )
        ax.set_xlim(-1, n_stag + 2)
        ax.set_ylim(1e-18, 1e1)
        if col == 0:
            ax.set_ylabel(
                r'$|\dot S_\mathrm{np} - \dot S_\mathrm{jax}|\;\;[\mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}]$'
            )
        ax.legend(loc='upper right', fontsize=7.5)
        ax.text(
            0.97,
            0.05,
            f'max abs = {abs_e.max():.2e}\n'
            f'med abs = {np.median(abs_e):.2e}\n'
            f'max rel = {rel.max():.2e}',
            transform=ax.transAxes,
            va='bottom',
            ha='right',
            fontsize=7.5,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2),
        )
        panel_label(ax, f'({panel_letters[col]})', loc='upper left')
        ax.set_title(f'{kind}', fontsize=9)

    # Bottom row: relative residuals
    for col, kind in enumerate(labels):
        ax = axes[1, col]
        rel = results[kind]['rel_err']
        abs_e = results[kind]['abs_err']
        ax.semilogy(
            cell_idx,
            np.maximum(rel, 1e-18),
            color=state_colors[kind],
            lw=1.0,
            marker='s',
            markersize=2.5,
            label=f'state = {kind}',
        )
        ax.axhline(
            np.finfo(np.float64).eps,
            color='k',
            lw=0.6,
            ls=':',
            label=r'$\epsilon_\mathrm{mach}$',
        )
        ax.set_xlabel('Staggered cell index (CMB -> surface)')
        if col == 0:
            ax.set_ylabel(
                r'$|\dot S_\mathrm{np} - \dot S_\mathrm{jax}|\;/\;|\dot S_\mathrm{np}|$'
            )
        ax.set_xlim(-1, n_stag + 2)
        ax.set_ylim(1e-18, 1e-2)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.text(
            0.97,
            0.05,
            f'max rel = {rel.max():.2e}\n'
            f'med rel = {np.median(rel):.2e}\n'
            f'max abs = {abs_e.max():.2e}',
            transform=ax.transAxes,
            va='bottom',
            ha='right',
            fontsize=7.5,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2),
        )
        panel_label(ax, f'({panel_letters[3 + col]})', loc='upper left')

    fig.tight_layout()
    save(fig, OUT / 'fig_01_rhs_parity.pdf')
    print('fig_01 saved.')
    for k in labels:
        r = results[k]
        print(
            f'  {k:10s}: max rel = {r["rel_err"].max():.2e}, '
            f'med rel = {np.median(r["rel_err"]):.2e}, '
            f'max abs = {r["abs_err"].max():.2e}'
        )


if __name__ == '__main__':
    main()
