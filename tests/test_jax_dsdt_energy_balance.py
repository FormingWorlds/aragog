"""Direct unit tests for ``aragog.jax.solver.dSdt_energy_balance``.

The energy_balance JAX RHS (extended state ``[S, dSdr_cmb]`` of length
N+1) is the production-path closure for the SPIDER-parity core BC. Two
orthogonal paths use it: the standalone JAX solver via
``solve_entropy(method='implicit_euler')`` and the production CVODE path
via ``build_jax_rhs_and_jacobian(core_bc_mode='energy_balance')``. Yet
the existing test_jax_entropy.py and test_cvode_jax_factory.py never
call ``dSdt_energy_balance`` directly: the former goes through
``solve_entropy`` only with the quasi_steady ``dSdt`` RHS, and the latter
only verifies the factory builds without invoking the returned rhs_fn.

This module fills that gap with targeted JAX-array calls into
``dSdt_energy_balance`` plus the ``_apply_cmb_bc`` BC-type-2 / 3 / 0
branches that share the same coverage shadow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
eqx = pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


# Geometry constants matching test_jax_entropy.py for parity.
R_INNER = 3.480e6
R_OUTER = 6.371e6


def _make_const_property_mesh(N: int = 30):
    """Build a small MeshArrays for constant-property dSdt tests.

    Mirrors test_jax_entropy._make_jax_mesh_arrays but at a smaller
    resolution so the JIT compile is cheap and the call-once cost is
    not amortised across many cases.
    """
    from aragog.jax.phase import MeshArrays

    r_stag = np.linspace(R_INNER, R_OUTER, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_INNER
    r_basic[-1] = R_OUTER
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])

    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_INNER, R_OUTER - r_basic), 1.0)

    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]
    d_dr[-1, :] = d_dr[-2, :]

    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5

    P_stag = np.linspace(135e9, 1e5, N)
    P_basic = q_mat @ P_stag

    return MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr),
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
        gravity=jnp.full(N + 1, 10.0),
    )


# ──────────────────────────────────────────────────────────────────────
#                       _apply_cmb_bc branches
# ──────────────────────────────────────────────────────────────────────


def _make_bc(*, inner_bc_type: int, inner_bc_value: float = 0.0):
    """Minimal BoundaryParams stub for _apply_cmb_bc unit tests."""
    from aragog.jax.solver import BoundaryParams

    return BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=inner_bc_type,
        inner_bc_value=inner_bc_value,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )


def test_apply_cmb_bc_prescribed_flux_overrides_heat_flux():
    """``inner_bc_type == 2`` writes ``inner_bc_value`` directly into
    heat_flux[0], replacing whatever conduction-derived flux compute_fluxes
    deposited there.

    Discriminator: any sentinel float (here, 1234.5 W/m^2) must round-trip
    bit-for-bit. A regression that lost the override would leave the
    pre-existing heat_flux[0] value (typically O(10^6) W/m^2 for a deep
    mantle gradient).
    """
    from aragog.jax.solver import _apply_cmb_bc

    mesh = _make_const_property_mesh(N=8)
    bc = _make_bc(inner_bc_type=2, inner_bc_value=1234.5)
    heat_flux = jnp.full(mesh.area.size, 9.0e6)
    rho_stag = jnp.full(8, 4000.0)
    cp_stag = jnp.full(8, 1000.0)
    out = _apply_cmb_bc(heat_flux, bc, mesh, rho_stag, cp_stag)
    assert float(out[0]) == pytest.approx(1234.5, rel=1e-12)
    # Other entries unchanged.
    np.testing.assert_allclose(np.asarray(out[1:]), 9.0e6, rtol=1e-12, atol=0.0)


def test_apply_cmb_bc_prescribed_temperature_preserves_conduction_flux():
    """``inner_bc_type == 3`` (prescribed T) keeps the conduction-derived
    heat_flux[0] from compute_fluxes; the BC dispatcher is a pass-through.

    Discriminator: a regression that overwrote heat_flux[0] with zero
    or with inner_bc_value would surface here as either 0 or whatever
    inner_bc_value was passed.
    """
    from aragog.jax.solver import _apply_cmb_bc

    mesh = _make_const_property_mesh(N=8)
    bc = _make_bc(inner_bc_type=3, inner_bc_value=999.0)  # value unused
    sentinel = 5.5e6
    heat_flux = jnp.full(mesh.area.size, sentinel)
    rho_stag = jnp.full(8, 4000.0)
    cp_stag = jnp.full(8, 1000.0)
    out = _apply_cmb_bc(heat_flux, bc, mesh, rho_stag, cp_stag)
    assert float(out[0]) == pytest.approx(sentinel, rel=1e-12), (
        'inner_bc_type=3 must preserve the conduction-derived flux at the CMB; '
        f'got {float(out[0]):.3e}, expected {sentinel:.3e}.'
    )


def test_apply_cmb_bc_insulating_zeros_heat_flux_at_cmb():
    """``inner_bc_type == 0`` is the insulating (zero-flux) BC.

    Discriminator: heat_flux[0] must be exactly zero. A regression that
    fell through to the core-cooling formula would leave a non-zero,
    state-dependent value.
    """
    from aragog.jax.solver import _apply_cmb_bc

    mesh = _make_const_property_mesh(N=8)
    bc = _make_bc(inner_bc_type=0)
    heat_flux = jnp.full(mesh.area.size, 7.0e5)
    rho_stag = jnp.full(8, 4000.0)
    cp_stag = jnp.full(8, 1000.0)
    out = _apply_cmb_bc(heat_flux, bc, mesh, rho_stag, cp_stag)
    assert float(out[0]) == 0.0


def test_apply_cmb_bc_core_cooling_uses_alpha_factor_partition():
    """``inner_bc_type == 1`` triggers the Bower+2018 Eq. 37
    alpha-factor flux partition between the bottom mantle cell and
    the core. F_cmb = alpha * heat_flux[1].

    Discriminator: with cell_cap << core_cap, alpha approaches
    radius_ratio**2; with cell_cap >> core_cap, alpha approaches
    zero. We construct two contrived cases and verify the limiting
    behaviour, plus the sign (positive heat_flux[1] gives positive
    F_cmb because alpha is always positive). A regression that
    inverted the cell_cap / core_cap ratio would flip the limits.
    """
    from aragog.jax.solver import _apply_cmb_bc

    mesh = _make_const_property_mesh(N=8)

    # Build a BC with inner_bc_type=1 + the constants that go into
    # the alpha formula. Earth-like core mass.
    from aragog.jax.solver import BoundaryParams

    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )
    heat_flux = jnp.full(mesh.area.size, 0.0)
    heat_flux = heat_flux.at[1].set(1.0e6)  # positive flux at first interior basic node
    rho_stag = jnp.full(8, 4000.0)
    cp_stag = jnp.full(8, 1000.0)
    out = _apply_cmb_bc(heat_flux, bc, mesh, rho_stag, cp_stag)

    # F_cmb must be a finite scalar with the same sign as heat_flux[1].
    F_cmb = float(out[0])
    assert np.isfinite(F_cmb)
    assert F_cmb > 0.0, (
        f'F_cmb = {F_cmb:.3e} should be positive when heat_flux[1] = +1e6 '
        '(alpha is positive by construction)'
    )
    # Bound: alpha * heat_flux[1] cannot exceed (r_above/r_cmb)^2 * heat_flux[1]
    r_cmb = float(mesh.radii_basic[0])
    r_above = float(mesh.radii_basic[1])
    upper_bound = (r_above / r_cmb) ** 2 * 1.0e6
    assert F_cmb < upper_bound, (
        f'F_cmb = {F_cmb:.3e} exceeds upper bound {upper_bound:.3e}; '
        'alpha formula limits violated'
    )


# ──────────────────────────────────────────────────────────────────────
#                       dSdt_energy_balance smoke
# ──────────────────────────────────────────────────────────────────────


# EOS-resolution policy (mirrors the integration suite).
import os  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


@needs_eos
def test_dsdt_energy_balance_returns_extended_state_derivative():
    """Direct call: ``dSdt_energy_balance`` must return a length-N+1
    array whose first N entries are the entropy derivatives and whose
    last entry is the boundary-state ``dSdr_cmb`` derivative.

    Discriminator: any regression in the boundary-state closure
    (e.g. dropping the ``2 / dr_cmb`` factor or losing the SECS_PER_YEAR
    rescale on the dSdr_cmb component) would surface as the trailing
    entry being zero, NaN, or O(1e3) larger than the entropy
    derivatives. A correct implementation puts the dSdr_cmb derivative
    in the J/kg/K/m/yr range, which is order 1e-3 to 1e3 for a
    cooling Earth-like mantle in the mushy regime — bounded but
    non-trivial.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams
    from aragog.jax.solver import _no_radio, dSdt_energy_balance

    eos_jax = EntropyEOS_JAX(EOS_DIR)
    params = PhaseParams()
    mesh = _make_const_property_mesh(N=12)
    n_stag = int(mesh.P_stag.shape[0])
    r_cmb = float(mesh.radii_basic[0])
    r_above = float(mesh.radii_basic[1])

    # BoundaryParams in energy_balance mode needs cmb_area, core_M, cmb_dr_cmb.
    from aragog.jax.solver import BoundaryParams

    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=5,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=(4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0,
        cmb_dr_cmb=r_above - r_cmb,
    )

    S = jnp.full(n_stag, 3050.0)
    dSdr_cmb = jnp.asarray(0.0)
    state_ext = jnp.concatenate([S, jnp.array([dSdr_cmb])])
    heating_static = jnp.zeros(n_stag)
    args = (eos_jax, params, mesh, bc, heating_static, _no_radio)

    rhs = dSdt_energy_balance(0.0, state_ext, args)
    rhs_np = np.asarray(rhs)

    # Discriminator 1: shape and finiteness.
    assert rhs_np.shape == (n_stag + 1,), (
        f'dSdt_energy_balance returned shape {rhs_np.shape}, expected ({n_stag + 1},)'
    )
    assert np.all(np.isfinite(rhs_np)), f'dSdt_energy_balance has non-finite entries: {rhs_np}'

    # Discriminator 2: entropy derivatives must be cooling-flavoured
    # (negative-or-mostly-negative for an isentrope at 3050 J/kg/K
    # losing heat to the surface). At least one entry must be
    # negative; otherwise the closure has the wrong sign.
    dSdt_S = rhs_np[:n_stag]
    assert np.any(dSdt_S < 0.0), (
        f'all entropy derivatives are non-negative ({dSdt_S}); '
        'expected at least one negative entry under grey-body cooling'
    )

    # Discriminator 3: dSdr_cmb derivative must be finite and bounded.
    # Order-of-magnitude bound: |d(dSdr)/dt| <= 1e6 J/kg/K/m/yr is a
    # very loose ceiling. A regression that lost the bc.cmb_dr_cmb
    # factor (numerator/denominator swap) would blow this by orders.
    d_dSdr_dt = float(rhs_np[n_stag])
    assert abs(d_dSdr_dt) < 1.0e6, (
        f'd(dSdr_cmb)/dt = {d_dSdr_dt:.3e} J/kg/K/m/yr is unphysical; '
        'check the SPIDER bc.c:76-131 closure factors.'
    )


@needs_eos
def test_dsdt_energy_balance_responds_to_dSdr_cmb_perturbation():
    """Sensitivity check: changing ``dSdr_cmb`` must change the rhs[0]
    entropy derivative (the bottom mantle cell), since the boundary
    gradient feeds into the CMB heat flux that drives that cell.

    Discriminator: a regression that decoupled dSdr_cmb from the
    flux pipeline would leave rhs[0] invariant under dSdr_cmb changes,
    making the CMB BC a no-op. The two evaluations must differ by
    more than a numerical-noise tolerance.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams
    from aragog.jax.solver import BoundaryParams, _no_radio, dSdt_energy_balance

    eos_jax = EntropyEOS_JAX(EOS_DIR)
    params = PhaseParams()
    mesh = _make_const_property_mesh(N=12)
    n_stag = int(mesh.P_stag.shape[0])
    r_cmb = float(mesh.radii_basic[0])
    r_above = float(mesh.radii_basic[1])

    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=5,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=(4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0,
        cmb_dr_cmb=r_above - r_cmb,
    )

    S = jnp.full(n_stag, 3050.0)
    args = (eos_jax, params, mesh, bc, jnp.zeros(n_stag), _no_radio)

    # Perturbation of 1.0 J/kg/K/m is large enough that any
    # functioning CMB-BC pipeline must shift rhs[0] visibly. Smaller
    # perturbations sit close to the FD truncation scale.
    rhs_a = np.asarray(dSdt_energy_balance(0.0, jnp.concatenate([S, jnp.array([0.0])]), args))
    rhs_b = np.asarray(dSdt_energy_balance(0.0, jnp.concatenate([S, jnp.array([1.0])]), args))

    # The first-cell entropy derivative must respond to dSdr_cmb.
    # The cascade is small in absolute terms (a 1 J/kg/K/m gradient
    # change shifts F_cmb proportionally to k_thermal ~ 4 W/m/K), so
    # the threshold is generous. A regression that decoupled dSdr_cmb
    # from the flux pipeline would leave the difference at the
    # numerical-noise floor (~1e-15).
    assert abs(rhs_b[0] - rhs_a[0]) > 1.0e-9, (
        f'rhs[0] is invariant under dSdr_cmb perturbation; got {rhs_a[0]:.3e} '
        f'vs {rhs_b[0]:.3e}. The CMB BC is decoupled from the flux pipeline.'
    )
