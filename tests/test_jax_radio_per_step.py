"""A2 tests: per-step radiogenic heating evaluated inside the JAX trace.

Locks the radio decay law of Soucasse §1.2

    H_radio_i(t) = heat_prod_i · abundance_i · concentration_i ·
                   exp(log(2) · (t0_i − t) / half_life_i)

into ``aragog.jax.solver.make_radio_heating_fn``, and verifies that
``dSdt`` actually consumes the time-dependent value (not a frozen
snapshot).

Discriminating values: each test picks t at 0, 1 half-life, and 2
half-lives so the expected ratio (1 : 0.5 : 0.25) distinguishes the
correct exponential decay from plausible wrong implementations
(linear interpolation, frozen value, missed log(2) factor).
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


def test_radio_heating_decays_to_half_at_one_half_life():
    """H_radio(t_half) = 0.5 · H_radio(0) for a single isotope.

    Discriminating: passes only for the exact log(2) exponential
    decay; would fail for linear-in-time, no-decay, or frozen-value
    implementations.
    """
    from aragog.jax.solver import make_radio_heating_fn

    hp = np.array([1.0e-9])  # 1 nW/kg power scale
    ab = np.array([1.0])
    cn = np.array([1.0])  # full mass fraction (test-only)
    t0 = np.array([0.0])
    hl = np.array([1.0e6])  # 1 Myr half-life

    H = make_radio_heating_fn(hp, ab, cn, t0, hl)

    H0 = float(H(0.0))
    H_half = float(H(1.0e6))
    H_two_half = float(H(2.0e6))

    assert H0 == pytest.approx(1.0e-9, rel=1e-12)
    assert H_half == pytest.approx(0.5e-9, rel=1e-10)
    assert H_two_half == pytest.approx(0.25e-9, rel=1e-10)


def test_radio_heating_sums_across_multiple_isotopes():
    """Total H = sum of per-isotope contributions.

    Discriminating: uses asymmetric per-isotope concentrations and
    half-lives so a wrong implementation that picks only one isotope
    or uses a mean-half-life would produce a different value.
    """
    from aragog.jax.solver import make_radio_heating_fn

    # Two isotopes, very different half-lives and concentrations
    hp = np.array([1.0e-9, 2.0e-9])
    ab = np.array([1.0, 1.0])
    cn = np.array([1.0, 0.5])
    t0 = np.array([0.0, 0.0])
    hl = np.array([1.0e6, 4.0e6])

    H = make_radio_heating_fn(hp, ab, cn, t0, hl)

    # At t=0: per-iso = (1e-9 · 1 · 1 · 1) + (2e-9 · 1 · 0.5 · 1) = 2e-9
    H0 = float(H(0.0))
    assert H0 == pytest.approx(2.0e-9, rel=1e-12)

    # At t = 1 Myr: iso0 halves to 0.5e-9; iso1 decays by 2^(-1/4) ≈ 0.8409
    H1 = float(H(1.0e6))
    expected = 1.0e-9 * 0.5 + 1.0e-9 * (2.0 ** (-0.25))
    assert H1 == pytest.approx(expected, rel=1e-10)


def test_radio_heating_zero_isotopes_returns_zero():
    """Empty list → 0.0 [W/kg], JAX-traceable scalar.

    Edge case used by the no-radio call path inside the production
    factory. Must return a finite zero (not NaN).
    """
    from aragog.jax.solver import _no_radio

    val = float(_no_radio(0.0))
    assert val == 0.0
    val_later = float(_no_radio(1.0e9))
    assert val_later == 0.0


def test_radio_heating_negative_time_inflates_correctly():
    """Time before t0 (e.g. accretion-phase IC) inflates H above t=t0.

    Edge case: setting t < t0 means we're looking at a time when
    more parent isotope existed. Verifies the sign of the exponent
    (would be inverted by a missing minus sign).
    """
    from aragog.jax.solver import make_radio_heating_fn

    hp = np.array([1.0e-9])
    ab = np.array([1.0])
    cn = np.array([1.0])
    t0 = np.array([0.0])
    hl = np.array([1.0e6])

    H = make_radio_heating_fn(hp, ab, cn, t0, hl)
    # At t = -1 Myr, half a half-life before t0, expect H = 2 · H(0)
    H_back = float(H(-1.0e6))
    assert H_back == pytest.approx(2.0e-9, rel=1e-10)


def test_radio_heating_unphysical_zero_half_life_does_not_crash():
    """Zero half-life is clamped by a numerical floor to avoid divide-by-zero.

    Physical edge case: a misconfigured half_life=0 input must not
    propagate as NaN/Inf into the integrator. The numerical floor
    inside ``make_radio_heating_fn`` handles it; this test locks
    the guard in place.
    """
    from aragog.jax.solver import make_radio_heating_fn

    hp = np.array([1.0e-9])
    ab = np.array([1.0])
    cn = np.array([1.0])
    t0 = np.array([0.0])
    hl = np.array([0.0])

    H = make_radio_heating_fn(hp, ab, cn, t0, hl)
    val = float(H(1.0e6))
    assert np.isfinite(val), 'H_radio with hl=0 must remain finite'


def test_radio_heating_short_half_life_decays_correctly():
    """Sub-year half-lives must follow physical decay, not be clamped.

    Pre-fix, the floor was 1.0 yr, which silently flattened the decay
    of any isotope with hl < 1 yr (none in production, but a latent
    correctness bug). The current floor of 1e-10 yr only guards the
    literal-zero divide while letting any physical isotope decay.

    Discriminating test: with hl = 0.5 yr (chosen well below the old
    1.0 yr floor) and t0 = 0, evaluate at t = 1.0 yr (= 2 half-lives).
    Correct result is H(t0)/4. Old (buggy) clamp would give
    H(t0)*exp(-LOG_TWO * 1.0 / 1.0) = H(t0)/2, which we explicitly
    reject below.
    """
    from aragog.jax.solver import make_radio_heating_fn

    hp = np.array([1.0e-9])
    ab = np.array([1.0])
    cn = np.array([1.0])
    t0 = np.array([0.0])
    hl = np.array([0.5])  # 0.5 yr, well below the historical 1.0 yr clamp

    H = make_radio_heating_fn(hp, ab, cn, t0, hl)
    H0 = float(H(0.0))  # heating at t = t0
    H_two_hl = float(H(1.0))  # heating at t = 2 * hl

    # After 2 half-lives, heating must be H(t0) / 4 to within float-64.
    np.testing.assert_allclose(H_two_hl, H0 / 4.0, rtol=1e-12)
    # The old (buggy) clamp would have given H(t0) / 2; reject that.
    assert abs(H_two_hl - H0 / 2.0) > 0.1 * H0, (
        'half-life floor too coarse: short-life isotope decay was clamped'
    )


@needs_eos
def test_dSdt_uses_live_radio_at_different_t():
    """dSdt at t=0 vs t=t_half_life must give different dS/dt.

    Pulls the same radio params through the production path
    (build_jax_rhs_and_jacobian → dSdt args tuple) and verifies that
    advancing time changes the resulting dS/dt. A frozen-radio bug
    (the pre-A2 behaviour) would give identical dS/dt at both times
    on the same entropy state.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import (
        BoundaryParams,
        dSdt,
        make_radio_heating_fn,
    )

    # Synthetic mesh covering the mushy band over an Earth-like depth.
    R_cmb, R_surf = 3.48e6, 6.371e6
    N = 24
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    P_stag = np.linspace(135.0e9, 1.0e5, N)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_basic = np.interp(r_basic, r_stag, P_stag)
    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_cmb, R_surf - r_basic), 1.0)

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

    g = 9.81
    mesh_jax = MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr_mat),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.full(r_basic.shape, g),
        gravity_stag=jnp.full(r_stag.shape, g),
    )

    eos_jax = EntropyEOS_JAX(EOS_DIR)

    # Mushy IC, modest gradient so the FD chain produces non-trivial
    # baseline dS/dt that the radio increment will perturb.
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
    params = PhaseParams(grav_sep=False, mixing=False, grain_size=0.1)

    # Strong, short-half-life isotope so the t=0 vs t=t_half effect
    # is large relative to ULP noise.
    H_radio = make_radio_heating_fn(
        np.array([1.0e-7]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([1.0e3]),
    )
    args0 = (eos_jax, params, mesh_jax, bc, jnp.zeros(N), H_radio)
    dsdt_t0 = np.asarray(dSdt(0.0, jnp.asarray(S_init), args0))
    dsdt_t_half = np.asarray(dSdt(1.0e3, jnp.asarray(S_init), args0))

    delta = dsdt_t_half - dsdt_t0
    # If dSdt froze radio at construction time, delta would be ~0
    # everywhere. Live evaluation at t=t_half_life should drop the
    # radio source by half, perturbing dS/dt by ~5e-8 W/kg / T which
    # is 1e-11..1e-10 J/kg/K/s, scaled by SECS_PER_YEAR (3.15e7) to
    # give 1e-4..1e-3 J/kg/K/yr -- well above ULP noise.
    max_abs_delta = float(np.max(np.abs(delta)))
    assert max_abs_delta > 1.0e-9, (
        f'dS/dt is essentially identical at t=0 and t=t_half_life '
        f'(max|Δ|={max_abs_delta:.3e}); H_radio is frozen, not live.'
    )
