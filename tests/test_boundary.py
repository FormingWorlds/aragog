"""Unit tests for ``aragog.solver.boundary.BoundaryConditions``.

The legacy T-form boundary dispatcher routes inner_boundary_condition
(CORE_BC) and outer_boundary_condition (SURFACE_BC) integer codes to
their physical implementations. The integration test harness exercises
it indirectly via full solver runs, but the dispatch logic itself
deserves a dedicated unit suite so that an unknown-code regression is
caught at parse time rather than mid-run.

State is mocked because the production State carries the entire
entropy/temperature buffer; for BC dispatch we only need ``heat_flux``,
``top_temperature`` and ``capacitance_staggered()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import constants as sp_constants

from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _ScalingsParameters,
    _SolverParameters,
)
from aragog.solver.boundary import BoundaryConditions

pytestmark = pytest.mark.unit


# ---- Fixtures: minimal Parameters / Mesh / State -----------------------------


def _build_parameters(
    *,
    inner_boundary_condition: int = 1,
    inner_boundary_value: float = 0.0,
    outer_boundary_condition: int = 1,
    outer_boundary_value: float = 1500.0,
    emissivity: float = 1.0,
    equilibrium_temperature: float = 255.0,
    core_heat_capacity: float = 880.0,
    tfac_core_avg: float = 1.147,
    param_utbl: bool = False,
    param_utbl_const: float = 1.0e-7,
) -> Parameters:
    """Build a Parameters instance with sensible defaults; only BC fields
    are exposed as kwargs for tests to flex.
    """
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=outer_boundary_condition,
        outer_boundary_value=outer_boundary_value,
        inner_boundary_condition=inner_boundary_condition,
        inner_boundary_value=inner_boundary_value,
        emissivity=emissivity,
        equilibrium_temperature=equilibrium_temperature,
        core_heat_capacity=core_heat_capacity,
        tfac_core_avg=tfac_core_avg,
        param_utbl=param_utbl,
        param_utbl_const=param_utbl_const,
    )
    return Parameters(
        boundary_conditions=bc,
        energy=_EnergyParameters(
            conduction=True,
            convection=False,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
        ),
        initial_condition=_InitialConditionParameters(),
        mesh=_MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=4,
            mixing_length_profile='nearest_boundary',
            core_density=10500.0,
        ),
        phase_solid=_PhaseParameters(
            density=4200.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=1e21,
        ),
        phase_liquid=_PhaseParameters(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=1.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=10.0,
        ),
        phase_mixed=_PhaseMixedParameters(
            latent_heat_of_fusion=4.0e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1.0e-3,
        ),
        radionuclides=[],
        scalings=_ScalingsParameters(),
        solver=_SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6),
    )


def _build_mock_mesh(
    *,
    cmb_radius: float = 3.480e6,
    above_radius: float = 4.45e6,
    cell_volume: float = 8.0e19,
    core_density: float = 10500.0,
) -> MagicMock:
    """Build a Mesh-like mock with only the attributes BoundaryConditions touches."""
    mesh = MagicMock()
    mesh.basic.radii = np.array([cmb_radius, above_radius, 5.4e6, 6.371e6])
    mesh.basic.volume = np.array([cell_volume, 1.0e20, 1.2e20])
    mesh.settings = SimpleNamespace(core_density=core_density)
    return mesh


def _build_mock_state(
    *,
    n_basic: int = 4,
    n_columns: int = 1,
    top_temperature: float = 3500.0,
    flux_above_cmb: float = 1.0e2,
    capacitance_staggered: float = 4.0e6,
):
    """Build a State-like mock with mutable heat_flux and the methods
    the BoundaryConditions class actually calls.
    """
    state = SimpleNamespace()
    state.heat_flux = np.zeros((n_basic, n_columns))
    state.heat_flux[1, :] = flux_above_cmb  # above-CMB flux for core_cooling
    state.top_temperature = np.array([top_temperature])
    # apply_flux_boundary_conditions logs state.temperature_basic at DEBUG;
    # python evaluates the arg even when the log filter drops it.
    state.temperature_basic = np.zeros((n_basic, n_columns))
    state.capacitance_staggered = lambda: np.full(n_basic - 1, capacitance_staggered)
    return state


# ---- apply_flux_inner_boundary_condition dispatch --------------------------


def test_inner_bc_type_2_writes_prescribed_heat_flux_to_cmb():
    """IBC=2 (prescribed flux): state.heat_flux[0, :] receives the
    parameter value verbatim. Use a discriminating non-zero value
    that doesn't equal any other defaulted field.
    """
    p = _build_parameters(
        inner_boundary_condition=2, inner_boundary_value=4.5e3, param_utbl_const=0.0
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    bc.apply_flux_inner_boundary_condition(state)
    assert float(state.heat_flux[0, 0]) == pytest.approx(4.5e3, rel=1e-12)


def test_inner_bc_type_3_is_pass_through():
    """IBC=3 (prescribed temperature) is a no-op for flux; CMB heat_flux
    must remain at its initialised zero. Discriminator: a regression
    that swapped IBC=3 to write inner_boundary_value into the flux
    field would silently inject a phantom temperature-as-flux.
    """
    p = _build_parameters(
        inner_boundary_condition=3, inner_boundary_value=4200.0, param_utbl_const=0.0
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    bc.apply_flux_inner_boundary_condition(state)
    assert float(state.heat_flux[0, 0]) == pytest.approx(0.0, abs=1e-30), (
        'IBC=3 wrote a non-zero flux at the CMB; the dispatch leaked.'
    )


def test_inner_bc_unknown_code_raises():
    """Unphysical IBC code 99 must raise ValueError, not silently fall
    through. Discriminator against losing the else-branch raise.
    """
    p = _build_parameters(
        inner_boundary_condition=99, inner_boundary_value=0.0, param_utbl_const=0.0
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    with pytest.raises(ValueError, match='inner_boundary_condition = 99'):
        bc.apply_flux_inner_boundary_condition(state)


# ---- apply_flux_outer_boundary_condition dispatch --------------------------


def test_outer_bc_type_4_writes_prescribed_heat_flux_to_surface():
    """OBC=4 (prescribed flux): state.heat_flux[-1, :] receives value."""
    p = _build_parameters(
        outer_boundary_condition=4, outer_boundary_value=2.5e5, param_utbl_const=0.0
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    bc.apply_flux_outer_boundary_condition(state)
    assert float(state.heat_flux[-1, 0]) == pytest.approx(2.5e5, rel=1e-12)


def test_outer_bc_type_5_is_pass_through():
    """OBC=5 (prescribed temperature): no flux assignment."""
    p = _build_parameters(
        outer_boundary_condition=5, outer_boundary_value=2000.0, param_utbl_const=0.0
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    bc.apply_flux_outer_boundary_condition(state)
    assert float(state.heat_flux[-1, 0]) == pytest.approx(0.0, abs=1e-30)


def test_outer_bc_type_2_zahnle_steam_raises_not_implemented():
    """OBC=2 (Zahnle steam atmosphere) is not implemented in Aragog;
    coupling is via PROTEUS. Must raise NotImplementedError, not pass.
    """
    p = _build_parameters(outer_boundary_condition=2, param_utbl_const=0.0)
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    with pytest.raises(NotImplementedError):
        bc.apply_flux_outer_boundary_condition(state)


def test_outer_bc_type_3_removed_raises_value_error():
    """OBC=3 (the legacy 'couple to atmodeller' slot) was removed; the
    real atmosphere coupling uses OBC=4 (prescribed surface heat flux).
    Setting OBC=3 must now hit the unknown-code branch.
    """
    p = _build_parameters(outer_boundary_condition=3, param_utbl_const=0.0)
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    with pytest.raises(ValueError, match='outer_boundary_condition = 3'):
        bc.apply_flux_outer_boundary_condition(state)


def test_outer_bc_unknown_code_raises_value_error():
    """OBC=99 must raise. Discriminator for the dispatch fallthrough."""
    p = _build_parameters(outer_boundary_condition=99, param_utbl_const=0.0)
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    with pytest.raises(ValueError, match='outer_boundary_condition = 99'):
        bc.apply_flux_outer_boundary_condition(state)


# ---- grey_body (OBC=1) -----------------------------------------------------


def test_grey_body_no_utbl_uses_top_temperature_directly():
    """OBC=1, param_utbl=False: F = epsilon * sigma * (T_top^4 - T_eq^4).

    Use T=3500 K, T_eq=255 K, epsilon=1: numerator difference is
    huge (≈ 1.5e14 K^4) so the result is dominated by T_top^4.
    Discriminator: the expected value distinguishes T^4 from T^3
    (would be a factor of 3500/4 too small).
    """
    p = _build_parameters(
        outer_boundary_condition=1,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        param_utbl=False,
        param_utbl_const=0.0,
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state(top_temperature=3500.0)
    bc.apply_flux_outer_boundary_condition(state)
    expected = sp_constants.Stefan_Boltzmann * (3500.0**4 - 255.0**4)
    assert float(state.heat_flux[-1, 0]) == pytest.approx(expected, rel=1e-12)


def test_grey_body_emissivity_scales_flux_linearly():
    """F is proportional to emissivity. With epsilon=0.7, the flux
    must be 0.7x the epsilon=1.0 value at the same T.

    Discriminator: catches a regression that absorbed emissivity into
    a missing ε^4 factor.
    """
    p_eps_1 = _build_parameters(
        outer_boundary_condition=1, emissivity=1.0, param_utbl_const=0.0
    )
    p_eps_07 = _build_parameters(
        outer_boundary_condition=1, emissivity=0.7, param_utbl_const=0.0
    )
    state_a = _build_mock_state(top_temperature=3500.0)
    state_b = _build_mock_state(top_temperature=3500.0)
    BoundaryConditions(p_eps_1, _build_mock_mesh()).apply_flux_outer_boundary_condition(state_a)
    BoundaryConditions(p_eps_07, _build_mock_mesh()).apply_flux_outer_boundary_condition(
        state_b
    )
    ratio = float(state_b.heat_flux[-1, 0]) / float(state_a.heat_flux[-1, 0])
    assert ratio == pytest.approx(0.7, rel=1e-12)


def test_grey_body_zero_when_top_temperature_equals_equilibrium():
    """Edge case: T_top == T_eq must give exactly zero net flux."""
    p = _build_parameters(
        outer_boundary_condition=1,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        param_utbl=False,
        param_utbl_const=0.0,
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state(top_temperature=255.0)
    bc.apply_flux_outer_boundary_condition(state)
    assert float(state.heat_flux[-1, 0]) == pytest.approx(0.0, abs=1e-9)


# ---- _utbl_tsurf (Cardano cubic) -------------------------------------------


def test_utbl_tsurf_returns_t_below_interior_when_b_positive():
    """The UTBL correction must produce T_surf < T_interior whenever
    b > 0 (the "cooling" direction). Use a strong b to make the
    drop substantial.

    Physical bound: T_surf in [0, T_interior]. Discriminator: a sign
    flip on Cardano (cbrt of wrong root) would give T_surf > T_interior.
    """
    p = _build_parameters(
        outer_boundary_condition=1,
        param_utbl=True,
        param_utbl_const=1.0e-9,  # strong cooling
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    t_int = np.array([3500.0])
    t_surf = bc._utbl_tsurf(t_int)
    assert float(t_surf[0]) < float(t_int[0]), (
        'UTBL surface temperature is not lower than interior; Cardano root selection is wrong.'
    )
    assert float(t_surf[0]) > 0.0, 'UTBL surface temperature is non-physical (negative).'


def test_utbl_tsurf_satisfies_cubic_equation():
    """Property: the returned T_surf must satisfy b*x^3 + x - T = 0.
    This is the strongest discriminator for Cardano correctness.
    """
    p = _build_parameters(param_utbl=True, param_utbl_const=2.0e-10)
    bc = BoundaryConditions(p, _build_mock_mesh())
    for t in [1500.0, 3000.0, 4500.0]:
        x = float(bc._utbl_tsurf(np.array([t]))[0])
        residual = 2.0e-10 * x**3 + x - t
        assert abs(residual) < 1.0e-6 * t, (
            f'Cardano residual {residual} too large for T={t} K, x={x}'
        )


def test_utbl_tsurf_array_input_returns_array():
    """Vectorised: array of interior T returns same-shape array."""
    p = _build_parameters(param_utbl=True, param_utbl_const=1.0e-10)
    bc = BoundaryConditions(p, _build_mock_mesh())
    t_int = np.array([1000.0, 2000.0, 3500.0])
    t_surf = bc._utbl_tsurf(t_int)
    assert t_surf.shape == (3,)
    # Property: monotonically increasing in t_int
    diffs = np.diff(t_surf)
    assert np.all(diffs > 0), 'UTBL surface T is not monotonic in interior T'


# ---- core_cooling (IBC=1) --------------------------------------------------


def test_core_cooling_writes_alpha_times_flux_above_to_cmb():
    """IBC=1 calls core_cooling, which writes
    state.heat_flux[0,:] = alpha * state.heat_flux[1,:].

    With reasonable Earth-mantle inputs alpha must be in (0, 1) and
    bounded above by (R_above/R_cmb)^2 = (4.45/3.48)^2 ≈ 1.63.

    Use a non-zero flux above the CMB to discriminate against a
    silent zeroing.
    """
    p = _build_parameters(
        inner_boundary_condition=1, inner_boundary_value=0.0, param_utbl_const=0.0
    )
    flux_above = 5.0e3
    state = _build_mock_state(flux_above_cmb=flux_above, capacitance_staggered=4.0e6)
    bc = BoundaryConditions(p, _build_mock_mesh())
    bc.apply_flux_inner_boundary_condition(state)
    cmb_flux = float(state.heat_flux[0, 0])
    radius_ratio_sq = (4.45e6 / 3.480e6) ** 2
    # Strict bound: alpha is in (0, ratio_sq).
    assert 0.0 < cmb_flux < radius_ratio_sq * flux_above, (
        f'core_cooling alpha out of expected range: '
        f'cmb={cmb_flux}, upper bound={radius_ratio_sq * flux_above}'
    )


def test_apply_temperature_bc_inner_3_writes_cmb_and_dTdr():
    """IBC=3 (prescribed temperature) writes inner_boundary_value
    to ``temperature_basic[0, :]`` and recomputes dTdr[0, :] using
    the centered-difference formula.

    Discriminator: the dTdr formula is
        dTdr[0] = 2 * (T_stag[0] - T_basic[0]) / delta_mesh[0] * dxidr[0]
    A regression that dropped the factor of 2 would still pass
    the temperature-set assertion but mis-set the gradient.
    """
    p = _build_parameters(
        inner_boundary_condition=3, inner_boundary_value=4200.0, param_utbl_const=0.0
    )
    mesh = MagicMock()
    mesh.basic.delta_mesh = np.array([1.0e5, 1.0e5, 1.0e5])
    mesh.dxidr = np.array([1.0, 1.0, 1.0, 1.0])
    bc = BoundaryConditions(p, mesh)

    n_basic, n_cols = 4, 1
    temperature = np.array([[3500.0], [3000.0], [2500.0], [2000.0]])
    temperature_basic = np.zeros((n_basic, n_cols))
    dTdr = np.zeros((n_basic, n_cols))
    bc.apply_temperature_boundary_conditions(temperature, temperature_basic, dTdr)

    assert temperature_basic[0, 0] == pytest.approx(4200.0, rel=1e-12)
    expected_dTdr = 2.0 * (3500.0 - 4200.0) / 1.0e5 * 1.0
    assert dTdr[0, 0] == pytest.approx(expected_dTdr, rel=1e-12)


def test_apply_temperature_bc_outer_5_writes_surface_and_dTdr():
    """OBC=5 (prescribed temperature) writes outer_boundary_value to
    ``temperature_basic[-1, :]`` and recomputes dTdr[-1, :] with the
    sign convention for the outer boundary (T_basic - T_stag, not the
    inner boundary's T_stag - T_basic).

    Discriminator: a regression that swapped the sign would invert
    the surface gradient (positive instead of negative).
    """
    p = _build_parameters(
        outer_boundary_condition=5, outer_boundary_value=2000.0, param_utbl_const=0.0
    )
    mesh = MagicMock()
    mesh.basic.delta_mesh = np.array([1.0e5, 1.0e5, 1.0e5])
    mesh.dxidr = np.array([1.0, 1.0, 1.0, 1.0])
    bc = BoundaryConditions(p, mesh)

    n_basic, n_cols = 4, 1
    temperature = np.array([[3500.0], [3000.0], [2500.0], [2200.0]])
    temperature_basic = np.zeros((n_basic, n_cols))
    dTdr = np.zeros((n_basic, n_cols))
    bc.apply_temperature_boundary_conditions(temperature, temperature_basic, dTdr)

    assert temperature_basic[-1, 0] == pytest.approx(2000.0, rel=1e-12)
    expected_dTdr = 2.0 * (2000.0 - 2200.0) / 1.0e5 * 1.0
    assert dTdr[-1, 0] == pytest.approx(expected_dTdr, rel=1e-12)


def test_apply_temperature_bc_other_codes_are_no_op():
    """Edge case: when both inner and outer BC codes are NOT 3 / 5
    respectively, the function leaves temperature_basic and dTdr
    unchanged.  Use IBC=1 / OBC=1 (the default flux modes).
    """
    p = _build_parameters(
        inner_boundary_condition=1,
        outer_boundary_condition=1,
        inner_boundary_value=0.0,
        param_utbl_const=0.0,
    )
    mesh = MagicMock()
    mesh.basic.delta_mesh = np.array([1.0e5, 1.0e5, 1.0e5])
    mesh.dxidr = np.array([1.0, 1.0, 1.0, 1.0])
    bc = BoundaryConditions(p, mesh)

    temperature = np.zeros((4, 1))
    temperature_basic = np.full((4, 1), 9999.0)
    dTdr = np.full((4, 1), 7777.0)
    bc.apply_temperature_boundary_conditions(temperature, temperature_basic, dTdr)
    np.testing.assert_array_equal(temperature_basic, np.full((4, 1), 9999.0))
    np.testing.assert_array_equal(dTdr, np.full((4, 1), 7777.0))


def test_apply_temperature_bc_melt_inner_3_recomputes_dphidr():
    """Mirror test for the melt-fraction BC dispatch: IBC=3 sets
    dphidr[0, :] from the melt-fraction interior values.
    """
    p = _build_parameters(
        inner_boundary_condition=3, inner_boundary_value=4200.0, param_utbl_const=0.0
    )
    mesh = MagicMock()
    mesh.basic.delta_mesh = np.array([1.0e5, 1.0e5, 1.0e5])
    mesh.dxidr = np.array([1.0, 1.0, 1.0, 1.0])
    bc = BoundaryConditions(p, mesh)

    melt_fraction = np.array([[0.7], [0.5], [0.3], [0.1]])
    melt_fraction_basic = np.array([[0.6], [0.4], [0.2], [0.05]])
    dphidr = np.zeros((4, 1))
    bc.apply_temperature_boundary_conditions_melt(melt_fraction, melt_fraction_basic, dphidr)
    expected = 2.0 * (0.7 - 0.6) / 1.0e5 * 1.0
    assert dphidr[0, 0] == pytest.approx(expected, rel=1e-12)


def test_apply_temperature_bc_melt_outer_5_recomputes_dphidr():
    """Mirror test for the melt-fraction BC dispatch: OBC=5 sets
    dphidr[-1, :] from melt_fraction interior values.
    """
    p = _build_parameters(
        outer_boundary_condition=5, outer_boundary_value=2000.0, param_utbl_const=0.0
    )
    mesh = MagicMock()
    mesh.basic.delta_mesh = np.array([1.0e5, 1.0e5, 1.0e5])
    mesh.dxidr = np.array([1.0, 1.0, 1.0, 1.0])
    bc = BoundaryConditions(p, mesh)

    melt_fraction = np.array([[0.7], [0.5], [0.3], [0.1]])
    melt_fraction_basic = np.array([[0.6], [0.4], [0.2], [0.05]])
    dphidr = np.zeros((4, 1))
    bc.apply_temperature_boundary_conditions_melt(melt_fraction, melt_fraction_basic, dphidr)
    expected = 2.0 * (0.05 - 0.1) / 1.0e5 * 1.0
    assert dphidr[-1, 0] == pytest.approx(expected, rel=1e-12)


def test_apply_flux_boundary_conditions_calls_both_dispatchers():
    """``apply_flux_boundary_conditions`` is the orchestrator that
    invokes inner + outer in order. Verify both happen by setting
    IBC=2 (writes to flux[0]) and OBC=4 (writes to flux[-1]) and
    confirming both sentinel values are present.
    """
    p = _build_parameters(
        inner_boundary_condition=2,
        inner_boundary_value=4.5e3,
        outer_boundary_condition=4,
        outer_boundary_value=2.5e5,
        param_utbl_const=0.0,
    )
    bc = BoundaryConditions(p, _build_mock_mesh())
    state = _build_mock_state()
    bc.apply_flux_boundary_conditions(state)
    assert float(state.heat_flux[0, 0]) == pytest.approx(4.5e3, rel=1e-12)
    assert float(state.heat_flux[-1, 0]) == pytest.approx(2.5e5, rel=1e-12)


def test_grey_body_with_utbl_correction_yields_lower_flux():
    """When ``param_utbl=True``, the surface T is reduced via Cardano,
    so the radiated flux (∝ T^4) must be lower than the no-UTBL
    baseline at the same interior temperature. Discriminator that
    catches a regression that wired up `_utbl_tsurf` but never
    consumed its output.
    """
    p_no_utbl = _build_parameters(
        outer_boundary_condition=1,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        param_utbl=False,
        param_utbl_const=0.0,
    )
    p_utbl = _build_parameters(
        outer_boundary_condition=1,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        param_utbl=True,
        param_utbl_const=1.0e-9,  # strong cooling
    )
    state_a = _build_mock_state(top_temperature=3500.0)
    state_b = _build_mock_state(top_temperature=3500.0)
    BoundaryConditions(p_no_utbl, _build_mock_mesh()).apply_flux_outer_boundary_condition(
        state_a
    )
    BoundaryConditions(p_utbl, _build_mock_mesh()).apply_flux_outer_boundary_condition(state_b)
    assert float(state_b.heat_flux[-1, 0]) < float(state_a.heat_flux[-1, 0]), (
        'UTBL grey-body radiated flux is not lower than no-UTBL baseline; '
        'the Cardano-corrected T_surf is not being used in the Stefan-Boltzmann formula.'
    )


def test_core_cooling_alpha_grows_when_core_capacity_dominates_cell_capacity():
    """Property: when C_core >> C_cell (large core_density and core
    heat capacity), alpha tends to (R_above/R_cmb)^2 — the geometric
    upper bound. Discriminator: an inverted ratio would push alpha
    toward 0 in this regime.
    """
    flux_above = 1.0e3
    # Configuration A: small core, default heat capacity.
    p_low = _build_parameters(
        inner_boundary_condition=1, core_heat_capacity=100.0, param_utbl_const=0.0
    )
    state_low = _build_mock_state(flux_above_cmb=flux_above, capacitance_staggered=1.0e7)
    BoundaryConditions(
        p_low, _build_mock_mesh(core_density=2000.0)
    ).apply_flux_inner_boundary_condition(state_low)
    # Configuration B: heavier core + larger heat capacity, fixed geometry.
    p_high = _build_parameters(
        inner_boundary_condition=1, core_heat_capacity=5000.0, param_utbl_const=0.0
    )
    state_high = _build_mock_state(flux_above_cmb=flux_above, capacitance_staggered=1.0e7)
    BoundaryConditions(
        p_high, _build_mock_mesh(core_density=15000.0)
    ).apply_flux_inner_boundary_condition(state_high)
    assert float(state_high.heat_flux[0, 0]) > float(state_low.heat_flux[0, 0]), (
        'alpha did not grow when core thermal capacity grew; '
        'Bower+2018 Eq. 37 ratio is inverted.'
    )
