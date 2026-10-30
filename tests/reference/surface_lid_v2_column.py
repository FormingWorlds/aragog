"""Solid shell for the V2 cross-check of the sub-grid lid against a resolved mesh.

A 25 km shell at a uniform 1300 K (below the solidus at every node), conduction only, with a
black-body surface radiating to 255 K. The resolved surface cools below 300 K, the floor of the
SPIDER test tables, so the runs use the PALEOS table extended to 67 K (``LOWT_EOS_MD5``). The
lid base is the 'fixed' rheological temperature of the shell, so the lid has no latent heat;
the 'local' stress closure keeps the lid from handing over to the mesh. The melt variant
(``melt_column``) starts the same shell in the mush, without rheology, so the lid base is the
solidus and the lid takes the latent heat of the melt it freezes.
"""

from __future__ import annotations

import numpy as np

T_I = 1300.0
LOWT_EOS_MD5 = '18d6f9eb5ba654d58fe3fad30ce118f0'  # temperature_solid.dat
TIMES_YR = np.geomspace(300.0, 5000.0, 8)
MELT_TIMES_YR = np.geomspace(100.0, 5000.0, 8)
PHI_0 = 0.2  # start entropy, as a fraction of the way from the solidus to the liquidus


def shell(
    n_nodes: int,
    lid: bool,
    eos,
    end_yr: float,
    start_yr: float = 0.0,
    T_eq: float = 255.0,
    rheology: bool = True,
):
    """The initialized solver of the shell (entropy not yet set); ``rheology=False`` turns the
    solid rheology off, which puts the lid base at the solidus."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_solver_bc_dispatch_smoke import _build

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=n_nodes, end_time=end_yr)
    p.solver.start_time = start_yr
    p.mesh.inner_radius = 6.346e6
    p.energy.convection = False
    p.boundary_conditions.equilibrium_temperature = T_eq
    p.boundary_conditions.subgrid_lid = lid
    p.boundary_conditions.normalize()
    p.phase_solid.rheology = SolidRheologyParams(
        enabled=rheology,
        stress_closure_mode='local',
        lid_base_mode='fixed',
        lid_base_temperature=T_I,
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    return s


def surface_flux(n_nodes: int, lid: bool, times_yr, eos) -> np.ndarray:
    """Surface heat flux [W/m^2] at ``times_yr`` from chained solves of the shell."""
    flux, y, t_prev = [], None, 0.0
    for t in times_yr:
        s = shell(n_nodes, lid, eos, float(t), t_prev)
        if y is None:
            S = [eos.entropy_at_temperature(float(P), T_I) for P in s._P_stag_flat]
        else:
            S = y[: s._n_stag]
            if lid:
                s.set_initial_lid(float(y[-2]), float(y[-1]))
        s.set_initial_entropy(np.asarray(S))
        s.solve()
        assert s._solution.status == 0
        flux.append(float(s.get_state().heat_flux[-1]))
        y, t_prev = np.asarray(s._solution.y[:, -1]), float(t)
    return np.array(flux)


def melt_column(n_nodes: int, lid: bool, times_yr, eos, perturb: float = 0.0):
    """Surface heat flux [W/m^2] and crust thickness [m] at ``times_yr`` from chained solves of
    the shell started at ``PHI_0`` of the way from the solidus to the liquidus entropy at each
    node (times ``1 + perturb``), without rheology. The crust is the lid thickness with the
    lid, else the depth where the entropy first reaches the solidus, linear between nodes."""
    flux, crust, y, t_prev = [], [], None, 0.0
    for t in times_yr:
        s = shell(n_nodes, lid, eos, float(t), t_prev, rheology=False)
        S_sol = np.asarray(eos.solidus_entropy(np.asarray(s._P_stag_flat)), dtype=float).ravel()
        if y is None:
            S_liq = np.asarray(eos.liquidus_entropy(np.asarray(s._P_stag_flat)), dtype=float)
            S = (S_sol + PHI_0 * (S_liq.ravel() - S_sol)) * (1.0 + perturb)
        else:
            S = y[: s._n_stag]
            if lid:
                s.set_initial_lid(float(y[-2]), float(y[-1]))
        s.set_initial_entropy(np.asarray(S))
        s.solve()
        assert s._solution.status == 0
        y, t_prev = np.asarray(s._solution.y[:, -1]), float(t)
        flux.append(float(s.get_state().heat_flux[-1]))
        mesh = s.evaluator.mesh
        z = (float(np.ravel(mesh.basic.radii)[-1]) - np.ravel(mesh.staggered.radii))[::-1]
        d = (y[: s._n_stag] - S_sol)[::-1]
        k = int(np.argmax(d >= 0.0))
        z_sol = z[k - 1] - d[k - 1] * (z[k] - z[k - 1]) / (d[k] - d[k - 1]) if k else np.nan
        crust.append(float(y[-2]) if lid else float(z_sol))
    return np.array(flux), np.array(crust)
