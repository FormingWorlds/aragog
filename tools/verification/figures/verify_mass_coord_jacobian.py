"""V&V Figure 7: Mass-coordinate transform Jacobian dxi/dr.

Verifies the mesh-internal mass-coordinate transform. The spatial
gradient of any quantity psi obeys

    d(psi)/dr = (rho*(r) / rho*_planet) * (r / xi)^2 * d(psi)/dxi,

so the geometric Jacobian is

    dxi/dr = (rho*(r) / rho*_planet) * (r / xi)^2,

with rho*(r) the local pseudo-density (Adams-Williamson EOS or
external eos_method=2 file) and rho*_planet the volume-averaged
planetary density. The aragog mesh adopts SPIDER's mass-coordinate
convention ``xi^3 = r_core^3 + 3 M_AW(r_core, r) / rho_avg_mantle``
so that

  (i)  xi(r_cmb) = r_core (xi_cmb pinned to the core radius).
  (ii) xi(r_top) = r_top (planetary radius), as the mantle-only
       volume-averaged density is the closing scaling.

The analytic Jacobian dxi/dr = (rho* / rho*_planet) (r/xi)^2 follows
directly from differentiating the integral form.

Panels:
  (a) dxi/dr along radius, computed two ways (mesh.dxidr and
      directly from the analytic formula).
  (b) Cumulative trapezoid integral of dxi/dr against the mesh's
      stored xi(r) - xi_cmb, showing pointwise agreement
      to within mesh truncation error.

Panels:
  (a) dxi/dr along the radius. Should be smooth and monotone in
      well-behaved cases (Adams-Williamson density rises with depth).
  (b) Numerical integral of dxi/dr against the analytical
      xi(r) - xi_cmb relation, showing pointwise agreement.

Run:
    conda activate proteus  # or your aragog env
    conda activate proteus
    python tools/verification/figures/verify_mass_coord_jacobian.py
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


def main():
    apply_rc()
    from z02_parity_multi_state import build_solver_and_jax_args  # noqa: WPS433

    solver, args, eos_jax = build_solver_and_jax_args()
    mesh = solver.evaluator.mesh

    # Pull spatial radii, mass radii, and per-node pseudo-density.
    r_basic   = np.asarray(mesh.basic.radii).ravel()        # spatial radii at basic nodes
    xi_basic  = np.asarray(mesh.basic.mass_radii).ravel()   # mass coord at basic nodes
    rho_basic = np.asarray(mesh.eos.basic_density).ravel()  # pseudo-density at basic nodes
    dxidr_arg = np.asarray(mesh.dxidr).ravel()              # transform Jacobian from mesh

    # Recompute dxi/dr from the analytic Jacobian directly. This must
    # match mesh.dxidr to numerical noise.
    rho_planet = float(getattr(mesh, '_planet_density'))
    dxidr_recompute = (rho_basic / rho_planet) * (r_basic / xi_basic) ** 2
    abs_err_jac = float(np.abs(dxidr_arg - dxidr_recompute).max())

    # Verify the transform invariants ─────────────────────────────────
    # SPIDER mass-coordinate convention used here: xi_cmb = r_core,
    # xi_top = r_top.
    inv_cmb = float(xi_basic[0]  - r_basic[0])
    inv_top = float(xi_basic[-1] - r_basic[-1])
    rho_core = float(getattr(mesh.settings, 'core_density', np.nan))
    # (iii) trapezoid integral of dxi/dr from r_cmb to r_top reproduces
    # xi_top - xi_cmb to within mesh truncation.
    xi_diff_integral = float(np.trapezoid(dxidr_arg, r_basic))
    xi_diff_analytic = float(xi_basic[-1] - xi_basic[0])
    integral_err = abs(xi_diff_integral - xi_diff_analytic)

    # Save raw arrays
    np.savez(
        DATA / 'fig_07_mass_coord_jacobian.npz',
        r_basic=r_basic, xi_basic=xi_basic,
        rho_basic=rho_basic, rho_planet=rho_planet,
        rho_core=rho_core,
        dxidr_mesh=dxidr_arg, dxidr_recompute=dxidr_recompute,
        abs_err_jac=abs_err_jac,
        inv_xi_cmb_minus_r_cmb=inv_cmb,
        inv_xi_top_minus_r_top=inv_top,
        xi_diff_integral=xi_diff_integral,
        xi_diff_analytic=xi_diff_analytic,
        integral_err=integral_err,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    depth_km = (r_basic[-1] - r_basic) / 1000.0

    # (a) dxi/dr along radius, with rho*/rho_planet overlay
    ax = axes[0]
    ax.plot(depth_km, dxidr_arg,
            color=PALETTE['numpy'], lw=1.6,
            label=r'$d\xi/dr$  from $\mathtt{mesh.dxidr}$')
    ax.plot(depth_km, dxidr_recompute,
            color=PALETTE['jax'], lw=0.9, ls='--',
            label=r'$(\rho^*/\rho^*_\mathrm{planet})\,(r/\xi)^2$  (analytic)')
    ax.invert_xaxis()
    ax.set_xlabel('Depth from surface (km)')
    ax.set_ylabel(r'$d\xi/dr$  (dimensionless)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('Transform Jacobian')
    ax.text(0.5, 0.05,
            f'$|\\mathtt{{mesh.dxidr}} - \\mathrm{{analytic}}|_\\infty$ = {abs_err_jac:.2e};  '
            fr'$\xi_\mathrm{{cmb}} - r_\mathrm{{cmb}} = {inv_cmb:+.1e}$ m;  '
            fr'$\xi_\mathrm{{top}} - r_\mathrm{{top}} = {inv_top:+.1e}$ m',
            transform=ax.transAxes, va='bottom', ha='center', fontsize=6.5,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2))
    panel_label(ax, '(a)', loc='upper left')

    # (b) Cumulative integral of dxi/dr vs analytic xi(r) - xi_cmb
    ax = axes[1]
    # numerical cumulative trapezoid of dxi/dr against r
    cumtrapz = np.zeros_like(r_basic)
    cumtrapz[1:] = np.cumsum(0.5 * (dxidr_arg[1:] + dxidr_arg[:-1])
                             * np.diff(r_basic))
    xi_minus_cmb_analytic = xi_basic - xi_basic[0]
    ax.plot(depth_km, xi_minus_cmb_analytic / 1000.0,
            color=PALETTE['analytic'], lw=2.0,
            label=r'$\xi(r) - \xi_\mathrm{cmb}$  (mesh)')
    ax.plot(depth_km, cumtrapz / 1000.0,
            color=PALETTE['jax'], lw=1.0, ls='--',
            label=r'$\int_{r_\mathrm{cmb}}^{r}\,d\xi/dr\,dr$  (trapezoid)')
    ax.invert_xaxis()
    ax.set_xlabel('Depth from surface (km)')
    ax.set_ylabel(r'$\xi - \xi_\mathrm{cmb}$  (km)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('Integral consistency')
    rel_int_err = abs((xi_diff_integral - xi_diff_analytic) / max(xi_diff_analytic, 1.0))
    ax.text(0.5, 0.05,
            f'trapezoid $-$ mesh $= {(xi_diff_integral - xi_diff_analytic):+.1e}$ m '
            f'({rel_int_err*100:.2f}% of mantle; '
            fr'$O(\Delta r^2)$, $N={len(r_basic)}$ cells)',
            transform=ax.transAxes, va='bottom', ha='center', fontsize=6.5,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, pad=2))
    panel_label(ax, '(b)', loc='upper left')

    fig.tight_layout()
    save(fig, OUT / 'fig_07_mass_coord_jacobian.pdf')
    print(f'fig_07 saved.')
    print(f'  rho_planet = {rho_planet:.3e} kg/m^3, rho_core = {rho_core:.3e} kg/m^3')
    print(f'  |mesh.dxidr - recompute| max = {abs_err_jac:.3e}')
    print(f'  invariants: xi_cmb-r_cmb = {inv_cmb:+.2e} m, xi_top-r_top = {inv_top:+.2e} m')
    print(f'  integral error = {(xi_diff_integral - xi_diff_analytic):.2e} m')


if __name__ == '__main__':
    main()
