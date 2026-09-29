"""The core exchanges energy with the mantle only through the CMB heat flux.

The quasi_steady core BC holds the bottom mantle cell and the core at one temperature. The
cell's internal heating must be shared by that lumped reservoir, not added to the core's
temperature on top of the flux it receives: at any state, M_core c_core tfac dT_core/dt must
equal -F_cmb A_cmb. For energy_balance the core is the CMB basic node, which the dSdr_cmb
equation ties to the same balance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

SECS_PER_YEAR = 3.15576e7
H_CELL = 5.0e-11  # W/kg
N_NODES = 40
H_PROFILE = np.linspace(3e-10, 1e-11, N_NODES - 1)  # W/kg per cell, bottom cell hottest


def _const_solver(core_bc, heating):
    """40-node const-properties mantle (T = T_ref exp((S - S_ref) / Cp)), core-cooling BC.

    ``heating`` is None, 'uniform' (H_CELL in every cell) or 'profile' (H_PROFILE).
    """
    from aragog.parser import _Radionuclide
    from aragog.solver.entropy_solver import EntropySolver

    p = _build_const_properties_parameters(n_nodes=N_NODES)
    bc = p.boundary_conditions
    bc.inner_boundary_condition, bc.core_bc = 1, core_bc
    bc.outer_boundary_condition, bc.outer_boundary_value = 4, 0.05
    p.energy.radionuclides = heating == 'uniform'
    # One isotope with an effectively infinite half-life: constant H_CELL.
    p.radionuclides = [_Radionuclide('X', 0.0, 1.0, 1.0, H_CELL, 1e20)] if heating else []
    p.energy.tidal = heating == 'profile'
    p.energy.tidal_array = H_PROFILE
    solver = EntropySolver(p, entropy_eos=None)
    solver.initialize()
    if core_bc == 'energy_balance':
        solver.set_initial_dSdr_cmb(-2.0e-5)  # entropy falls outward: F_cmb > 0
    r = np.asarray(solver._r_stag_flat)
    solver.set_initial_entropy(3000.0 + 20.0 * (r[-1] - r) / (r[-1] - r[0]))
    return solver


def _expected_heating(heating):
    return {None: 0.0, 'uniform': H_CELL, 'profile': H_PROFILE}[heating]


@pytest.mark.unit
@pytest.mark.parametrize('heating', [None, 'uniform', 'profile'])
@pytest.mark.parametrize('core_bc', ['quasi_steady', 'energy_balance'])
def test_core_temperature_follows_the_cmb_flux(core_bc, heating):
    """numpy RHS: the core's temperature rate is -F_cmb A_cmb / C', heating on or off."""
    solver = _const_solver(core_bc, heating)
    n = solver._n_stag
    dy = solver._dSdt_single(0.0, np.asarray(solver._S0, dtype=float).ravel()) / SECS_PER_YEAR
    H = np.asarray(solver.state.heating).ravel()
    assert H == pytest.approx(_expected_heating(heating), rel=1e-12)
    cp = 1000.0  # const_Cp: dT/dS = T / Cp
    if core_bc == 'quasi_steady':
        T = float(np.asarray(solver.state.phase_staggered.temperature()).flat[0])
        dS_core = dy[0]
    else:
        T = float(np.asarray(solver.state.phase_basic.temperature()).flat[0])
        dS_core = dy[0] - solver._cmb_dr_half * dy[n]
    core_rate = T / cp * dS_core
    balance = (
        -solver.state._heat_flux[0] * solver._cmb_area / (solver._core_cap * solver._core_tfac)
    )
    assert balance != 0.0
    assert core_rate == pytest.approx(balance, rel=1e-9, abs=0.0)


@pytest.mark.unit
@pytest.mark.parametrize('heating', ['uniform', 'profile'])
def test_lumped_reservoir_conserves_energy(heating):
    """numpy quasi_steady: mantle plus core gain the total heating minus the surface outflow."""
    solver = _const_solver('quasi_steady', heating)
    n = solver._n_stag
    dy = solver._dSdt_single(0.0, np.asarray(solver._S0, dtype=float).ravel())[:n]
    dy /= SECS_PER_YEAR
    rho, cp = 4000.0, 1000.0  # const_rho, const_Cp
    T = np.asarray(solver.state.phase_staggered.temperature()).ravel()
    m = rho * np.asarray(solver._volume_flat)
    C_core = solver._core_cap * solver._core_tfac
    gain = np.sum(m * T * dy) + C_core * T[0] / cp * dy[0]
    H = _expected_heating(heating)
    expected = -solver.state._heat_flux[-1] * solver._area_flat[-1] + np.sum(H * m)
    # Interior powers (up to ~1e24 W) cancel in the sum; bound round-off by their scale.
    scale = np.abs(solver.state._heat_flux * solver._area_flat).max()
    assert gain == pytest.approx(expected, rel=0.0, abs=1e-14 * scale)


@pytest.mark.unit
def test_jax_lumped_partition_takes_the_cell_heating_from_the_outflow():
    """JAX _apply_cmb_bc, type 1: F_cmb A_cmb (1 + C_cell / C') = F_1 A_1 - Q_0."""
    jnp = pytest.importorskip('jax.numpy')
    from aragog.jax.solver import _apply_cmb_bc
    from tests.test_jax_dsdt_energy_balance import _make_bc, _make_const_property_mesh

    mesh, bc = _make_const_property_mesh(N=8), _make_bc(inner_bc_type=1)
    rho, cp, F1 = 4000.0, 1000.0, 0.05
    r_cmb, vol, area = float(mesh.radii_basic[0]), float(mesh.volume[0]), np.asarray(mesh.area)
    H = 2.0 * F1 * area[1] / (rho * vol)  # Q_0 = 2 F_1 A_1: both terms count, F_cmb < 0
    heat_flux = jnp.zeros(mesh.area.size).at[1].set(F1)
    F_cmb = float(_apply_cmb_bc(heat_flux, bc, mesh, jnp.full(8, rho), jnp.full(8, cp), H)[0])
    C_core = 4.0 / 3.0 * np.pi * r_cmb**3 * bc.core_density * bc.core_heat_capacity
    C_core *= bc.tfac_core_avg
    lhs = F_cmb * area[0] * (1.0 + vol * rho * cp / C_core)
    assert lhs == pytest.approx(F1 * area[1] - H * rho * vol, rel=1e-9, abs=0.0)
    assert F_cmb < 0.0


@pytest.mark.unit
def test_jax_dsdt_conserves_energy_with_a_heating_profile(monkeypatch):
    """JAX dSdt, quasi_steady, no EOS tables: mantle plus core gain the total heating minus
    the surface outflow, with the bottom cell heated more than the rest."""
    jnp = pytest.importorskip('jax.numpy')
    import aragog.jax.solver as js
    from aragog.jax.phase import FluxOutput, PhaseParams, PhaseProperties
    from tests.test_jax_dsdt_energy_balance import _make_const_property_mesh

    N, rho, cp = 8, 4000.0, 1000.0
    mesh = _make_const_property_mesh(N=N)

    def phase(eos, params, P, S):
        T = 2000.0 * jnp.exp((S - 3000.0) / cp)
        one = jnp.ones_like(S)
        return PhaseProperties(
            T,
            rho * one,
            cp * one,
            *(0.0 * one,) * 3,
            1.0 * one,
            *(0.0 * one,) * 3,
            rho * T,
            0.0 * one,
            0.0 * one,
            one,
        )

    def fluxes(S, t, eos, params, mesh, heating):
        z = jnp.zeros(N + 1)
        return FluxOutput(jnp.linspace(0.08, 0.02, N + 1), z, z, heating, z)

    monkeypatch.setattr(js, 'evaluate_phase', phase)
    monkeypatch.setattr(js, 'compute_fluxes', fluxes)
    bc = js.BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.05,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )
    H = np.linspace(3e-10, 1e-11, N)
    S = jnp.linspace(3100.0, 3000.0, N)
    args = (None, PhaseParams(), mesh, bc, jnp.asarray(H), js._no_radio)
    dS = np.asarray(js.dSdt(0.0, S, args)) / SECS_PER_YEAR
    T = np.asarray(phase(None, None, None, S).temperature)
    m = rho * np.asarray(mesh.volume)
    r_cmb = float(mesh.radii_basic[0])
    C_core = 4.0 / 3.0 * np.pi * r_cmb**3 * bc.core_density * bc.core_heat_capacity
    C_core *= bc.tfac_core_avg
    gain = np.sum(m * T * dS) + C_core * T[0] / cp * dS[0]
    expected = -0.05 * float(mesh.area[-1]) + np.sum(H * m)
    assert gain == pytest.approx(expected, rel=1e-9, abs=0.0)


@pytest.mark.smoke
def test_jax_core_temperature_follows_the_cmb_flux(monkeypatch):
    """JAX quasi_steady RHS: C' dT_core/dt = -F_cmb A_cmb with the bottom cell heated."""
    from tests.test_entropy_solver_option_z_smoke import EOS_DIR, _build_parameters

    if not EOS_DIR.exists():
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}')
    jnp = pytest.importorskip('jax.numpy')
    import aragog.jax.solver as js
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams, evaluate_phase
    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver(_build_parameters(n_nodes=12), entropy_eos=EntropyEOS(EOS_DIR))
    solver.initialize()
    mesh = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
    eos, params = EntropyEOS_JAX(EOS_DIR), PhaseParams()
    bc = js.BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.05,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )
    n = mesh.P_stag.shape[0]
    S = jnp.linspace(3100.0, 3000.0, n)
    flux, real = {}, js._apply_cmb_bc
    monkeypatch.setattr(
        js, '_apply_cmb_bc', lambda *a, **k: flux.setdefault('F', real(*a, **k))
    )
    args = (eos, params, mesh, bc, jnp.full(n, H_CELL), js._no_radio)
    dS = np.asarray(js.dSdt(0.0, S, args)) / SECS_PER_YEAR
    phase = evaluate_phase(eos, params, mesh.P_stag, S)
    T, cp = float(phase.temperature[0]), float(phase.heat_capacity[0])
    r_cmb = float(mesh.radii_basic[0])
    C_core = 4.0 / 3.0 * np.pi * r_cmb**3 * bc.core_density * bc.core_heat_capacity
    C_core *= bc.tfac_core_avg
    balance = -float(flux['F'][0]) * float(mesh.area[0]) / C_core
    assert T / cp * dS[0] == pytest.approx(balance, rel=1e-9, abs=0.0)
