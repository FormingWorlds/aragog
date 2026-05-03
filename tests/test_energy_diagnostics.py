"""Tests for the EOS-consistent enthalpy table and total_enthalpy diagnostic.

Validates the new energy-conservation infrastructure added in commit A1:
``EntropyEOS._build_enthalpy_table`` / ``specific_enthalpy`` /
``_specific_enthalpy_scalar`` and ``aragog.output.diagnostics.total_enthalpy``.

These tests target the rigorous replacement for the legacy
``E_th = sum(m * Cp_apparent * T)`` proxy, which is unusable for
conservation because the latent-heat-blended ``Cp_apparent`` spikes through
the rheological transition. The new EOS-consistent enthalpy uses the path
integral of ``dh = T dS + (1/rho) dP`` from a fixed table-corner anchor,
so latent heat enters automatically through the entropy jump across the
mushy zone at constant melting temperature.

Requires SPIDER-format P-S tables; skipped if the test fixture directory
is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

EOS_DIR = Path(
    os.environ.get(
        'ARAGOG_TEST_EOS_DIR',
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    )
)

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def eos():
    """Load the EntropyEOS once per module (the h-table build takes a few seconds)."""
    from aragog.eos import EntropyEOS

    return EntropyEOS(EOS_DIR)


# ── Anchor and table structure ──────────────────────────────────────────


@needs_eos
def test_anchor_h_is_zero_by_construction(eos):
    """At the table corner (P_min, S_min) the integration anchor must be exactly zero.

    This is the definition of the enthalpy zero-point; any non-zero value here
    would mean the cumulative integral is being primed with a non-zero seed.
    """
    h_anchor = float(eos.specific_enthalpy(eos.P_min, eos.S_min))
    assert h_anchor == 0.0, (
        f'Anchor enthalpy must be 0 by construction; got {h_anchor:.3e} J/kg'
    )


@needs_eos
def test_h_table_has_no_nans(eos):
    """The precomputed h table must be NaN-free across its entire grid."""
    h = eos._h_grid
    n_nan = int(np.isnan(h).sum())
    assert n_nan == 0, f'h_grid has {n_nan} NaNs out of {h.size}'
    assert h.shape[0] == len(eos._h_grid_P)
    assert h.shape[1] == len(eos._h_grid_S)


@needs_eos
def test_h_table_grid_covers_full_S_range(eos):
    """The h-table S grid must span at least from solidus-anchor to high-melt
    (validates the union-S-grid fix; using only the melt table's S grid
    silently clamped low-S queries to a single value)."""
    S_grid = eos._h_grid_S
    assert S_grid[0] <= eos.S_min + 1.0, (
        f'h-table S grid starts at {S_grid[0]:.0f}, but EOS S_min is {eos.S_min:.0f}'
    )
    assert S_grid[-1] >= eos.S_max - 1.0


# ── Property-based: thermodynamic correctness ──────────────────────────


@needs_eos
@pytest.mark.parametrize('P', [1.0e9, 5.0e10, 1.0e11, 2.0e11])
def test_enthalpy_strictly_increases_with_S_at_fixed_P(eos, P):
    """Since dh = T dS + (1/rho) dP, at fixed P we must have dh/dS = T > 0
    everywhere in the table (T is in Kelvin so always positive). Distinguishes
    the correct integral from common bugs: missing the cumsum (gives constant
    h), wrong sign (gives decreasing h), or integrating dT instead of T (gives
    non-monotonic h with sign flips at extrema)."""
    Ss = np.linspace(eos.S_min + 50.0, eos.S_max - 50.0, 12)
    hs = eos.specific_enthalpy(np.full_like(Ss, P), Ss)
    diffs = np.diff(hs)
    assert np.all(diffs > 0), (
        f'At P={P:.0e} Pa, enthalpy must be strictly increasing in S; '
        f'got {(diffs <= 0).sum()} non-positive diffs out of {len(diffs)}'
    )


@needs_eos
def test_latent_heat_captured_across_mushy_zone(eos):
    """Crossing the mushy zone at constant P, the entropy jump (S_liq - S_sol)
    occurs at near-constant temperature T_phase, so dh = T_phase * dS_phase
    should approximately equal the local latent heat per unit mass.

    This is the discriminating test for whether the table integration
    captures the latent-heat contribution to enthalpy automatically (the
    motivating reason for choosing the entropy-form integral over a
    sensible-Cp formula). A bug that drops the mushy-zone temperature
    weighting would underestimate the jump by ~50%.
    """
    P_test = 5.0e10  # mid-mantle
    S_sol = float(eos.solidus_entropy(P_test))
    S_liq = float(eos.liquidus_entropy(P_test))
    L = float(eos.latent_heat(P_test))
    h_sol = float(eos.specific_enthalpy(P_test, S_sol))
    h_liq = float(eos.specific_enthalpy(P_test, S_liq))
    dh_observed = h_liq - h_sol
    # Sanity: dh should be of order L_fusion (a few MJ/kg), and bounded
    # above by T_liq * (S_liq - S_sol) which is the path-integral upper
    # bound when T does not drop in the mushy interior.
    T_liq = float(eos.temperature(P_test, S_liq))
    upper_bound = T_liq * (S_liq - S_sol)
    assert dh_observed > 0.5 * L, (
        f'Mushy-zone dh = {dh_observed:.2e} J/kg but L_fusion = {L:.2e}; '
        f'latent-heat contribution looks suppressed'
    )
    assert dh_observed <= 1.05 * upper_bound, (
        f'Mushy-zone dh = {dh_observed:.2e} exceeds T_liq*(S_liq-S_sol) = '
        f'{upper_bound:.2e}; integration is overshooting'
    )


@needs_eos
def test_dh_matches_trapezoidal_estimate_in_single_phase(eos):
    """Within a single phase region (where T(S) is smooth), the table
    integral dh = h(S_b) - h(S_a) at fixed P must agree with a 2-point
    trapezoidal T-average estimate to a few percent. This is the
    ground-truth check that the cumulative integral closes correctly.

    Test interval is chosen above the liquidus to stay in fully-melt
    phase; the mushy zone deliberately produces a larger discrepancy
    because T plateaus at the melting curve, which a 2-point estimate
    over-predicts.
    """
    P = 1.0e11
    S_liq = float(eos.liquidus_entropy(P))
    S_a = S_liq + 200.0
    S_b = S_liq + 1500.0
    # Confirm we are in fully-melt phase across the interval
    phi_a = float(eos.melt_fraction(P, S_a))
    phi_b = float(eos.melt_fraction(P, S_b))
    assert phi_a >= 0.99 and phi_b >= 0.99, (
        f'Test interval should be all-melt; got phi=({phi_a:.3f}, {phi_b:.3f})'
    )
    h_a = float(eos.specific_enthalpy(P, S_a))
    h_b = float(eos.specific_enthalpy(P, S_b))
    dh = h_b - h_a
    T_a = float(eos.temperature(P, S_a))
    T_b = float(eos.temperature(P, S_b))
    estimate = 0.5 * (T_a + T_b) * (S_b - S_a)
    rel = abs(dh - estimate) / max(abs(estimate), 1.0)
    assert rel < 0.05, (
        f'dh = {dh:.3e} J/kg vs trapezoidal estimate = {estimate:.3e}; '
        f'relative discrepancy {rel:.1%} > 5% (single-phase region '
        f'should integrate cleanly)'
    )


# ── Edge cases and numerical safety ────────────────────────────────────


@needs_eos
def test_out_of_range_low_S_clamps_without_nan(eos):
    """Querying below the table S_min must return a finite value (the
    boundary h), not NaN or an exception. Cells in solid-only regimes
    can have entropy below the lowest tabulated value when initial
    conditions are off-table; the diagnostic must degrade gracefully.
    """
    P = 1.0e10
    h_below = float(eos.specific_enthalpy(P, eos.S_min - 5000.0))
    h_at_min = float(eos.specific_enthalpy(P, eos.S_min))
    assert np.isfinite(h_below)
    assert h_below == pytest.approx(h_at_min, rel=1e-12)


@needs_eos
def test_out_of_range_high_S_clamps_without_nan(eos):
    """Same for super-high entropy queries (vapor / superheated melt)."""
    P = 1.0e10
    h_above = float(eos.specific_enthalpy(P, eos.S_max + 5000.0))
    h_at_max = float(eos.specific_enthalpy(P, eos.S_max))
    assert np.isfinite(h_above)
    assert h_above == pytest.approx(h_at_max, rel=1e-12)


@needs_eos
def test_negative_pressure_clamps_without_nan(eos):
    """Negative pressure is unphysical but the lookup must clamp rather
    than crash, so a malformed solver state never produces a NaN that
    propagates into the helpfile."""
    h = float(eos.specific_enthalpy(-1.0e5, 5000.0))
    h_at_zero = float(eos.specific_enthalpy(eos.P_min, 5000.0))
    assert np.isfinite(h)
    assert h == pytest.approx(h_at_zero, rel=1e-12)


@needs_eos
def test_scalar_path_matches_vectorized(eos):
    """The pure-Python ``_specific_enthalpy_scalar`` and the vectorized
    ``specific_enthalpy`` must produce identical values to within rounding
    on a sample of mixed in-range and out-of-range points.
    """
    test_pts = [
        (eos.P_min, eos.S_min),  # anchor
        (eos.P_max, eos.S_max),  # opposite corner
        (5.0e10, 4500.0),  # mid-mantle, mushy
        (1.0e11, 7500.0),  # deep, melt
        (1.0e9, 500.0),  # near-surface, cold solid
        (-1.0e5, 50.0),  # below table on both axes
    ]
    for P, S in test_pts:
        h_vec = float(eos.specific_enthalpy(P, S))
        h_scl = eos._specific_enthalpy_scalar(P, S)
        assert h_vec == pytest.approx(h_scl, rel=1e-9, abs=1e-3), (
            f'Scalar/vectorized mismatch at (P={P:.2e}, S={S:.0f}): '
            f'vec={h_vec:.6e}, scl={h_scl:.6e}'
        )


# ── total_enthalpy diagnostic ──────────────────────────────────────────


@needs_eos
def test_total_enthalpy_uniform_mantle(eos):
    """For a synthetic mantle where every cell sees the same (P, S),
    total_enthalpy must equal M_total * h_lookup(P, S). Distinguishes
    correct mass-weighting from common bugs: volume-weighting (factor of
    rho off), per-cell instead of total integral (factor of N off),
    or unmixed phase blending (jump at solidus crossing).
    """
    from aragog.output.diagnostics import total_enthalpy

    n_cells = 25
    P_uniform = 7.5e10
    S_uniform = 4200.0
    M_per_cell = 8.0e22
    P_stag = np.full(n_cells, P_uniform)
    S_stag = np.full(n_cells, S_uniform)
    mass_stag = np.full(n_cells, M_per_cell)
    E_total = total_enthalpy(eos, P_stag, S_stag, mass_stag)
    E_expected = float(eos.specific_enthalpy(P_uniform, S_uniform)) * M_per_cell * n_cells
    assert E_total == pytest.approx(E_expected, rel=1e-10)


@needs_eos
def test_total_enthalpy_zero_mass_cells_dont_contribute(eos):
    """Cells with zero mass must contribute exactly zero (edge case for
    the moving-mesh case where some shells can collapse)."""
    from aragog.output.diagnostics import total_enthalpy

    P_stag = np.array([1e10, 5e10, 1e11, 2e11])
    S_stag = np.array([3500.0, 4500.0, 5500.0, 6500.0])
    mass_stag_with_zero = np.array([1e22, 0.0, 1e22, 0.0])
    mass_stag_only_nonzero = np.array([1e22, 1e22])
    E_with_zero = total_enthalpy(eos, P_stag, S_stag, mass_stag_with_zero)
    E_only_nz = total_enthalpy(
        eos,
        np.array([1e10, 1e11]),
        np.array([3500.0, 5500.0]),
        mass_stag_only_nonzero,
    )
    assert E_with_zero == pytest.approx(E_only_nz, rel=1e-12)


@needs_eos
def test_total_enthalpy_partition_invariant(eos):
    """Splitting one cell of mass M into two cells of mass M/2 with the
    same (P, S) must give exactly the same total enthalpy. This pins
    down that total_enthalpy is a true integral, not an averaging
    artifact."""
    from aragog.output.diagnostics import total_enthalpy

    P, S, M = 8.0e10, 4800.0, 1.5e23
    E_one = total_enthalpy(eos, np.array([P]), np.array([S]), np.array([M]))
    E_two = total_enthalpy(
        eos,
        np.array([P, P]),
        np.array([S, S]),
        np.array([M / 2.0, M / 2.0]),
    )
    assert E_one == pytest.approx(E_two, rel=1e-12)


@needs_eos
def test_total_enthalpy_mass_scaling_linearity(eos):
    """E_state must be exactly linear in cell mass (h is intensive, mass
    is extensive). Multiplying every mass by a positive factor must
    multiply E_state by the same factor."""
    from aragog.output.diagnostics import total_enthalpy

    rng = np.random.default_rng(42)
    n = 20
    P_stag = rng.uniform(1e9, 2e11, n)
    S_stag = rng.uniform(3500.0, 6500.0, n)
    mass_stag = rng.uniform(1e22, 1e23, n)
    E_1x = total_enthalpy(eos, P_stag, S_stag, mass_stag)
    factor = 3.7
    E_kx = total_enthalpy(eos, P_stag, S_stag, mass_stag * factor)
    assert E_kx == pytest.approx(E_1x * factor, rel=1e-12)


# ── Per-source heating decomposition (commit A2) ───────────────────────


def _make_minimal_state(
    eos, *, radio=False, dilatation=False, tidal=False, tidal_array=None, radionuclides_obj=None
):
    """Build a minimal EntropyState wired to a test mesh and EOS for
    direct ``update()`` calls in source-decomposition tests.

    Mirrors the helper in tests/test_entropy_verification.py with the
    additional source flags exposed so each per-source heating array
    can be tested in isolation.
    """
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    N = 30
    R_cmb, R_surf = 3.48e6, 6.371e6
    P_cmb, P_surf = 1.35e11, 1.0e5
    r_stag = np.linspace(R_cmb, R_surf, N)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_stag = np.linspace(P_cmb, P_surf, N)
    P_basic = np.interp(r_basic, r_stag, P_stag)

    class Mesh:
        pass

    class SubMesh:
        pass

    mesh = Mesh()
    mesh.basic = SubMesh()
    mesh.staggered = SubMesh()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.minimum(r_basic - R_cmb, R_surf - r_basic)
    mesh.basic.mixing_length = np.maximum(ml, 1.0)
    mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
    mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3
    mesh.basic.pressure = P_basic
    mesh.staggered.pressure = P_stag
    mesh.basic.mass_radii = r_basic
    mesh.staggered.mass_radii = r_stag
    mesh.dxidr = np.ones_like(r_basic)
    dr = np.diff(r_stag)

    def quantity_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[0], out[-1] = q[0], q[-1]
        out[1:-1] = 0.5 * (q[:-1] + q[1:])
        return out

    def quantity_at_staggered_nodes(q):
        q = np.asarray(q).flatten()
        return 0.5 * (q[:-1] + q[1:])

    def d_dr_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[1:-1] = np.diff(q) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.quantity_at_staggered_nodes = quantity_at_staggered_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes

    phase_stag = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(P_stag)
    phase_basic = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(P_basic)

    class Eval:
        pass

    evaluator = Eval()
    evaluator.mesh = mesh
    if radio and radionuclides_obj is not None:
        evaluator.radionuclides = [radionuclides_obj]

    return EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=radio,
        dilatation=dilatation,
        tidal=tidal,
        tidal_array=tidal_array,
    )


@needs_eos
def test_per_source_heating_arrays_exist_and_sum_to_total(eos):
    """The decomposition must close: heating_radio + heating_dil +
    heating_tidal == heating, at every staggered node, after a single
    update() call. Catches the obvious bug where one source is added
    to the cumulative ``_heating`` but not stashed into its per-source
    array (or vice versa).
    """

    class StubRadio:
        def get_heating(self, t_yr):
            return 1.5e-9  # W/kg, near Earth's mean specific decay heat

    state = _make_minimal_state(
        eos,
        radio=True,
        tidal=True,
        tidal_array=[3.0e-10],
        radionuclides_obj=StubRadio(),
    )
    S0 = np.full(30, 4500.0)
    state.update(S0, time=4.567)
    decomposed = state.heating_radio + state.heating_dil + state.heating_tidal
    np.testing.assert_allclose(
        decomposed,
        state.heating,
        rtol=0,
        atol=1e-20,
        err_msg='per-source heating arrays do not sum to cumulative _heating',
    )


@needs_eos
def test_disabled_sources_have_exactly_zero_heating_arrays(eos):
    """When a source flag is off, its per-source array must be exactly
    zero everywhere -- not "small", not "ones plus epsilon". A leak of
    even 1e-18 W/kg across all cells in a 5e24 kg mantle is 5e6 W of
    fictitious power that would corrupt the long-time conservation
    diagnostic.
    """
    state = _make_minimal_state(eos)  # all sources disabled by default
    S0 = np.full(30, 5500.0)
    state.update(S0, time=4.567)
    assert np.all(state.heating_radio == 0.0)
    assert np.all(state.heating_dil == 0.0)
    assert np.all(state.heating_tidal == 0.0)
    assert np.all(state.heating == 0.0), 'All source flags off: cumulative heating must be zero'


@needs_eos
def test_radio_heating_uniform_per_kg(eos):
    """Radiogenic heating per unit mass is uniform across the mantle
    (decay rate per kg of bulk silicate is independent of P and T to
    first order). The per-source array must reflect this; a bug that
    weights radio heating by, say, density would silently break this.
    """
    radio_per_kg = 2.5e-12

    class StubRadio:
        def get_heating(self, t_yr):
            return radio_per_kg

    state = _make_minimal_state(eos, radio=True, radionuclides_obj=StubRadio())
    S0 = np.full(30, 4500.0)
    state.update(S0, time=4.567)
    np.testing.assert_allclose(
        state.heating_radio,
        radio_per_kg,
        rtol=1e-12,
        err_msg='radio heating array should be uniform = get_heating(t)',
    )


@needs_eos
def test_tidal_heating_with_array_input(eos):
    """When tidal_array has length n_staggered, the per-source array
    must equal that input element-wise; when length 1, all cells get
    the scalar value. Tests both the array and the broadcast paths.
    """
    n = 30
    tidal_profile = np.linspace(1e-12, 1e-11, n).tolist()
    state = _make_minimal_state(eos, tidal=True, tidal_array=tidal_profile)
    state.update(np.full(n, 4500.0), time=4.567)
    np.testing.assert_allclose(
        state.heating_tidal,
        tidal_profile,
        rtol=1e-12,
        err_msg='tidal heating per-cell profile not preserved',
    )

    # Scalar broadcast path
    tidal_scalar = 5.0e-11
    state_b = _make_minimal_state(eos, tidal=True, tidal_array=[tidal_scalar])
    state_b.update(np.full(n, 4500.0), time=4.567)
    np.testing.assert_allclose(
        state_b.heating_tidal,
        tidal_scalar,
        rtol=1e-12,
        err_msg='tidal scalar must broadcast uniformly',
    )


@needs_eos
def test_unphysical_negative_radio_heating_is_passed_through(eos):
    """A radionuclides stub returning a negative get_heating() value
    (unphysical, but possible for a malformed config) must be reflected
    in heating_radio without silent clamping. Diagnostics should expose
    the bad input rather than mask it. The cumulative heating array
    must still equal the sum of the per-source arrays.
    """

    class BadRadio:
        def get_heating(self, t_yr):
            return -1.0e-10  # unphysical negative power

    state = _make_minimal_state(eos, radio=True, radionuclides_obj=BadRadio())
    S0 = np.full(30, 4500.0)
    state.update(S0, time=0.5)
    assert np.all(state.heating_radio < 0.0), (
        'negative get_heating must propagate into heating_radio for diagnosis'
    )
    np.testing.assert_allclose(
        state.heating_radio + state.heating_dil + state.heating_tidal,
        state.heating,
        atol=1e-20,
    )
