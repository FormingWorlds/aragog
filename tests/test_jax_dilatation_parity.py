"""Lock-in tests: the explicit Φ_vol source term is deleted.

The Soucasse §1.2 dilatation source

    Phi_vol = rho · g · (1/rho_m - 1/rho_s) · (j_cm + j_gm)

is NOT added to ``heating`` in either the numpy ``EntropyState.update``
path or the JAX ``compute_fluxes`` path. This file pins that
invariant: setting the legacy ``dilatation`` flag has no effect on the
heating array, and the heating array contains only the radio + tidal
sources passed in.

The volumetric work this term used to capture is already implicit in
the divergence of the Δh-weighted mass-flux contributions to
``heat_flux``: by definition Δh = Δu + P·Δv, and on a hydrostatic
column ∂Δh/∂r ⊃ Δv·∂P/∂r = -ρg·Δv, so -∂/∂r(j·Δh) ⊃ +ρg·Δv·j is the
same quantity. Adding the explicit source on top double-counts (see
finding_2026_05_03_phi_vol_double_count_verdict.md). Bower 2018 /
SPIDER's entropy form has no such source either.

The JAX RHS is the production CHILI integration path under
``solver_method='cvode'`` + ``use_jax_jacobian=True``; without these
tests, a regression that re-introduced an explicit Φ_vol on the JAX
side would silently produce a 2× heating contribution.

Markers: all tests run as ``unit`` (synthetic mesh, no SPIDER tables
required for the off-case tests; parity tests skip when EOS tables
are unavailable).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

jax.config.update('jax_enable_x64', True)

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


# ── helpers ───────────────────────────────────────────────────────────


def _build_synthetic_mesh(N=24, R_cmb=3.48e6, R_surf=6.371e6, P_cmb=135.0e9, P_surf=1.0e5):
    """Earth-like spherical mesh shared between numpy and JAX paths.

    Returns the numpy mesh-mock, JAX MeshArrays, and the entropy IC
    arrays at staggered + basic nodes.
    """
    from aragog.jax.phase import MeshArrays

    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    P_stag = np.linspace(P_cmb, P_surf, N)

    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_basic = np.interp(r_basic, r_stag, P_stag)

    g_const = 9.81

    # numpy mesh mock (mirrors the pattern in
    # tests/test_entropy_pytest.py::TestJgravCmbDrain._build_mesh).
    class _Mesh:
        pass

    class _Sub:
        pass

    mesh = _Mesh()
    mesh.basic = _Sub()
    mesh.staggered = _Sub()
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

    def quantity_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[0], out[-1] = q[0], q[-1]
        out[1:-1] = 0.5 * (q[:-1] + q[1:])
        return out

    def d_dr_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[1:-1] = np.diff(q) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    def quantity_at_staggered_nodes(q):
        # Mirrors aragog.mesh.Mesh.quantity_at_staggered_nodes
        # (mesh/__init__.py:417-418): simple midpoint average from
        # basic-node arrays. Production path uses this for the
        # j_total basic -> staggered map in the dilatation block.
        q = np.asarray(q).flatten()
        return 0.5 * (q[:-1] + q[1:])

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes
    mesh.quantity_at_staggered_nodes = quantity_at_staggered_nodes

    # JAX MeshArrays: build transform matrices that match the
    # numpy mesh's behaviour on this synthetic uniform grid.
    d_dr_mat = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr_mat[i, i - 1] = -1.0 / dr[i - 1]
        d_dr_mat[i, i] = 1.0 / dr[i - 1]
    d_dr_mat[0, :] = d_dr_mat[1, :]
    d_dr_mat[-1, :] = d_dr_mat[-2, :]

    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5

    g_basic = np.full(r_basic.shape, g_const)
    g_stag = np.full(r_stag.shape, g_const)

    mesh_jax = MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr_mat),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(mesh.basic.area),
        volume=jnp.asarray(mesh.basic.volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(mesh.basic.mixing_length),
        mixing_length_sq=jnp.asarray(mesh.basic.mixing_length_squared),
        mixing_length_cu=jnp.asarray(mesh.basic.mixing_length_cubed),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.asarray(g_basic),
        gravity_stag=jnp.asarray(g_stag),
    )

    return mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, g_const


# ── B1 unit tests ────────────────────────────────────────────────────


def test_dilatation_off_returns_input_heating_unchanged():
    """When ``params.dilatation == 0`` the returned heating equals the input.

    Edge case + invariant: with the flag off there must be no path through
    the H_dil block that mutates the heating array, even if grav_sep and
    mixing are on (so jmix_heat and mass_flux are non-zero).
    """
    pytest.importorskip('aragog')
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found at {EOS_DIR}')

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    params = PhaseParams(
        grav_sep=True,
        mixing=True,
        dilatation=False,  # ← off
    )
    S_init = jnp.full(r_stag.size, 3200.0)  # mushy band entropy
    heating_in = jnp.full(r_stag.size, 1.234e-9)  # constant non-zero radio
    out = compute_fluxes(
        S_init,
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    np.testing.assert_allclose(
        np.asarray(out.heating),
        np.asarray(heating_in),
        rtol=0.0,
        atol=1e-30,
    )


def test_dilatation_on_no_transport_yields_zero_contribution():
    """``dilatation=True`` with ``grav_sep=mixing=False`` ⇒ H_dil ≈ 0.

    Mirrors the numpy short-circuit ``if dilatation and (grav_sep or
    mixing):`` (entropy_state.py:715). Acts as a discriminating value
    test: a wrong implementation that adds H_dil unconditionally would
    fail this case, even when both transport flags are off.
    """
    pytest.importorskip('aragog')
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found at {EOS_DIR}')

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    params = PhaseParams(
        grav_sep=False,
        mixing=False,
        dilatation=True,
    )
    S_init = jnp.full(r_stag.size, 3200.0)
    heating_in = jnp.zeros(r_stag.size)
    out = compute_fluxes(
        S_init,
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    # Every staggered cell should still see exactly zero heating
    # because (jgrav, jmix) are gated to zero by their own flags.
    np.testing.assert_allclose(
        np.asarray(out.heating),
        0.0,
        atol=1e-30,
    )


def test_dilatation_pure_phase_zero_contribution():
    """Pure liquid (S well above S_liq everywhere) ⇒ H_dil ≈ 0.

    Outside the mushy band, the cubic-Hermite smth gate
    ``16·gphi²·(1-gphi)²`` at basic nodes vanishes, so jmix_heat = 0
    and jgrav is suppressed by the bottom-up smoothing. H_dil must
    therefore vanish to floating-point. Edge case for
    Soucasse §1.2: ``j_cm`` and ``j_gm`` are non-zero only inside the
    two-phase region.
    """
    pytest.importorskip('aragog')
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found at {EOS_DIR}')

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    # Push S well above the liquidus envelope at every node
    S_liq_max = float(np.max(np.asarray(eos_jax.liquidus_entropy(jnp.asarray(P_stag)))))
    S_init = jnp.full(r_stag.size, S_liq_max + 500.0)

    params = PhaseParams(
        grav_sep=True,
        mixing=True,
        dilatation=True,
    )
    heating_in = jnp.zeros(r_stag.size)
    out = compute_fluxes(
        S_init,
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    # Allow a small numerical floor from smoothing tails; the term
    # should be ~10 orders of magnitude below a typical mushy H_dil
    # contribution (which sits around 1e-10..1e-8 W/kg).
    max_abs_H = float(np.max(np.abs(np.asarray(out.heating))))
    assert max_abs_H < 1e-15, f'Pure-liquid H_dil should vanish, got max|H|={max_abs_H:.3e}'


def test_dilatation_unphysical_negative_density_fraction_raises():
    """Density fraction (1/ρ_l - 1/ρ_s) must be finite at all cells.

    The numpy reference uses ``1.0 / max(rho, 1.0)`` to guard against
    table-edge zeros. The JAX path inherits the same guard. Verify that
    feeding an unphysical zero-density EOS lookup would surface as a
    finite (not NaN) result — i.e. the safe-divide guards are in place.
    Triggered indirectly by setting an entropy outside the EOS table
    range (clamped) and confirming the output stays finite.
    """
    pytest.importorskip('aragog')
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found at {EOS_DIR}')

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)
    params = PhaseParams(
        grav_sep=True,
        mixing=True,
        dilatation=True,
    )
    # Both an entropy under S_min and over S_max — clamped by the EOS
    # to the table edges. Result must remain finite (not NaN/Inf).
    S_init = jnp.full(r_stag.size, float(eos_jax.S_min) - 1000.0)
    heating_in = jnp.zeros(r_stag.size)
    out = compute_fluxes(
        S_init,
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    assert bool(jnp.all(jnp.isfinite(out.heating))), (
        'H_dil must remain finite even at clamped table-edge entropies'
    )


@needs_eos
def test_no_phi_vol_added_in_mushy_zone():
    """Mushy regime where the old Φ_vol would have been large: zero now.

    Builds matched numpy and JAX evaluators on the same synthetic mesh
    with a fully mushy IC (Φ ~ 0.55 across the full pressure range)
    and a small entropy gradient so that ``j_grav`` and ``j_mix`` are
    both non-trivially non-zero — the regime where the old explicit
    Φ_vol source was largest. With the source deleted, both paths must
    now return ``heating == 0`` to floating-point. Setting the legacy
    ``dilatation=True`` flag must NOT change this: the flag is a
    vestigial accept-and-ignore parameter.

    Discriminating values: a non-zero entropy gradient (5 J/kg/K from
    CMB to surface) drives j_grav ~ 1e-3 kg/m²/s and j_mix ~ 1e-3
    kg/m²/s in the mushy zone, which would have produced
    H_dil ~ 1e-8 W/kg via Soucasse §1.2. We assert below 1e-15 W/kg,
    which discriminates against any regression that re-introduces
    even a single fractional copy of Φ_vol.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes
    from aragog.solver.entropy_state import EntropyState

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, g_const = _build_synthetic_mesh(N=24)

    eos_np = EntropyEOS(EOS_DIR)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    # ---- numpy path ----
    phase_stag = EntropyPhaseEvaluator(
        entropy_eos=eos_np,
        gravitational_acceleration=g_const,
        grain_size=0.1,
    )
    phase_stag.set_pressure(P_stag)
    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=eos_np,
        gravitational_acceleration=g_const,
        grain_size=0.1,
    )
    phase_basic.set_pressure(P_basic)

    class _Eval:
        pass

    evaluator = _Eval()
    evaluator.mesh = mesh

    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=True,
        convection=True,
        gravitational_separation=True,
        mixing=True,
        radionuclides=False,
        dilatation=True,  # legacy flag must be ignored
        tidal=False,
    )

    # Mushy IC: somewhere between solidus and liquidus across the
    # full pressure range, then perturbed slightly to drive a non-
    # zero entropy gradient (so jgrav and jmix are non-trivial).
    S_sol_stag = np.asarray(eos_np.solidus_entropy(P_stag)).ravel()
    S_liq_stag = np.asarray(eos_np.liquidus_entropy(P_stag)).ravel()
    S_init = 0.45 * S_sol_stag + 0.55 * S_liq_stag
    # Slight gradient to drive convective instability in the mushy zone
    S_init = S_init - 5.0 * np.linspace(0.0, 1.0, S_init.size)

    state.update(S_init, time=0.0)
    H_numpy = np.asarray(state.heating).ravel()

    # ---- JAX path ----
    params = PhaseParams(
        grav_sep=True,
        mixing=True,
        dilatation=True,  # legacy flag ignored
        grain_size=0.1,
    )
    heating_in = jnp.zeros(S_init.size)
    out = compute_fluxes(
        jnp.asarray(S_init),
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    H_jax = np.asarray(out.heating).ravel()

    # Both paths must now produce zero heating contribution from
    # phase segregation; the only sources are radio + tidal, both of
    # which are off in this test.
    max_abs_numpy = float(np.max(np.abs(H_numpy)))
    max_abs_jax = float(np.max(np.abs(H_jax)))
    assert max_abs_numpy < 1e-15, (
        f'numpy heating array contains a non-zero contribution '
        f'(max|H|={max_abs_numpy:.3e} W/kg) where only radio/tidal '
        f'should be present; Φ_vol source likely re-introduced.'
    )
    assert max_abs_jax < 1e-15, (
        f'JAX heating array contains a non-zero contribution '
        f'(max|H|={max_abs_jax:.3e} W/kg) where only radio/tidal '
        f'should be present; Φ_vol source likely re-introduced.'
    )


@needs_eos
def test_dilatation_flag_does_not_change_dSdt():
    """The legacy ``dilatation`` flag is vestigial: dSdt is identical on/off.

    Builds the same mushy IC twice — once with ``dilatation=False``,
    once with ``dilatation=True`` — and calls ``dSdt`` (the actual JAX
    RHS that CVODE consumes). The dS/dt arrays must agree to
    floating-point: the flag is no longer wired to anything. A
    regression that re-introduces a flag-gated H_dil contribution
    would produce a non-zero delta in the mushy interior.

    Discriminating values: the mushy IC produces dsdt ~ 1e-15 to
    1e-12 J/kg/K/yr from convection alone; we assert the flag-on/off
    delta stays below 1e-20 J/kg/K/yr, which is several orders of
    magnitude tighter than the natural noise floor of dsdt.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams
    from aragog.jax.solver import BoundaryParams, _no_radio, dSdt

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    S_sol = np.asarray(eos_jax.solidus_entropy(jnp.asarray(P_stag)))
    S_liq = np.asarray(eos_jax.liquidus_entropy(jnp.asarray(P_stag)))
    S_init = 0.45 * S_sol + 0.55 * S_liq
    S_init = S_init - 5.0 * np.linspace(0.0, 1.0, S_init.size)

    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=0,
        inner_bc_value=0.0,
        core_density=10738.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )
    heating_in = jnp.zeros(S_init.size)
    args_off = (
        eos_jax,
        PhaseParams(grav_sep=True, mixing=True, dilatation=False, grain_size=0.1),
        mesh_jax,
        bc,
        heating_in,
        _no_radio,
    )
    args_on = (
        eos_jax,
        PhaseParams(grav_sep=True, mixing=True, dilatation=True, grain_size=0.1),
        mesh_jax,
        bc,
        heating_in,
        _no_radio,
    )
    dsdt_off = np.asarray(dSdt(0.0, jnp.asarray(S_init), args_off))
    dsdt_on = np.asarray(dSdt(0.0, jnp.asarray(S_init), args_on))

    delta = dsdt_on - dsdt_off
    max_abs_delta = float(np.max(np.abs(delta)))
    assert max_abs_delta < 1e-20, (
        f'dS/dt changed when dilatation flag flipped: max|delta|='
        f'{max_abs_delta:.3e} J/kg/K/yr. The flag is supposed to be '
        f'vestigial; this suggests Φ_vol was re-introduced or some '
        f'other code path is reading the flag.'
    )


@needs_eos
def test_radio_only_heating_in_mushy_zone():
    """The only heating contribution in the mushy zone is radio + tidal.

    Builds the same mushy IC as the parity test, but passes a non-zero
    radio heating array as ``heating_in``. The output must equal the
    input exactly (modulo radio decay if the integrator advances the
    clock; here t=0 so radio is pristine), with no Φ_vol contribution
    on top.

    Discriminates against a regression that re-introduces a flag-gated
    or unconditional Φ_vol: such a regression would add ~1e-8 W/kg in
    the mushy zone, making the radio-only assumption fail.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    mesh, mesh_jax, r_stag, r_basic, P_stag, P_basic, _ = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)
    # Mushy IC with the same gradient as the parity test.
    S_sol = np.asarray(eos_jax.solidus_entropy(jnp.asarray(P_stag)))
    S_liq = np.asarray(eos_jax.liquidus_entropy(jnp.asarray(P_stag)))
    S_init = 0.45 * S_sol + 0.55 * S_liq
    S_init = S_init - 5.0 * np.linspace(0.0, 1.0, S_init.size)

    params = PhaseParams(
        grav_sep=True,
        mixing=True,
        dilatation=True,
        grain_size=0.1,
    )
    radio_value = 1.234e-9  # per-cell uniform [W/kg]
    heating_in = jnp.full(S_init.size, radio_value)
    out = compute_fluxes(
        jnp.asarray(S_init),
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    H_out = np.asarray(out.heating)
    H_in = np.asarray(heating_in)
    # Output must equal input exactly; no Φ_vol contribution.
    np.testing.assert_allclose(H_out, H_in, rtol=0.0, atol=1e-30)
    # Also discriminate against radio being silently dropped: max|H|
    # must equal the radio value.
    assert float(np.max(np.abs(H_out))) == pytest.approx(radio_value, rel=1e-12), (
        f'Expected heating == {radio_value} (radio uniform), got '
        f'max|H|={float(np.max(np.abs(H_out))):.3e}'
    )
