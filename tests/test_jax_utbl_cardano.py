"""Numpy-vs-JAX parity for the UTBL Cardano surface BC correction (U1).

Bower+2018 Eq. 18 reduces the surface radiating temperature to account
for an unresolved ultra-thin thermal boundary layer at the magma-ocean
surface. The numpy path (solver/boundary.py:_utbl_tsurf) implements
the Cardano cubic root for
    b * x^3 + x - T_interior = 0
with b = param_utbl_const. The JAX path (jax/solver.py:_utbl_tsurf_jax)
must produce byte-identical results so that PROTEUS+JAX runs with
param_utbl=True match SPIDER's UTBL-corrected surface flux.

Production configs use param_utbl=False, so this divergence
does not currently affect any paper-line run. The tests pin parity
against future SPIDER-parity test configs that may flip the flag.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from aragog.jax.solver import (  # noqa: E402
    BoundaryParams,
    _apply_surface_bc,
    _utbl_tsurf_jax,
)


@pytest.mark.unit
@pytest.mark.parametrize('T_interior', [1500.0, 2500.0, 3500.0, 4500.0])
@pytest.mark.parametrize('b', [1e-7, 1e-6, 1e-5])
def test_cardano_jax_vs_numpy_parity(T_interior, b):
    """JAX Cardano root must match the numpy formula to machine precision.

    Both implement the analytical Cardano formula for
    b * x^3 + x - T = 0; the only difference is jnp vs np.
    """
    p = 1.0 / b
    q = -T_interior / b
    discriminant = q**2 / 4.0 + p**3 / 27.0
    sqrt_disc = np.sqrt(discriminant)
    T_surf_numpy = np.cbrt(-q / 2.0 + sqrt_disc) + np.cbrt(-q / 2.0 - sqrt_disc)

    T_surf_jax = float(_utbl_tsurf_jax(jnp.asarray(T_interior), jnp.asarray(b)))

    assert T_surf_jax == pytest.approx(T_surf_numpy, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('T_interior', [2000.0, 3000.0, 4000.0])
def test_cardano_recovers_input_when_b_negligible(T_interior):
    """When b -> 0, T_surf -> T_interior."""
    b = 1e-15
    T_surf = float(_utbl_tsurf_jax(jnp.asarray(T_interior), jnp.asarray(b)))
    assert T_surf == pytest.approx(T_interior, rel=1e-6)


@pytest.mark.unit
def test_cardano_satisfies_cubic():
    """The returned T_surf must satisfy b * T_surf^3 + T_surf = T_interior."""
    b = 1e-7
    T_interior = 3500.0
    T_surf = float(_utbl_tsurf_jax(jnp.asarray(T_interior), jnp.asarray(b)))
    residual = b * T_surf**3 + T_surf - T_interior
    assert abs(residual) < 1e-6 * T_interior


@pytest.mark.unit
def test_cardano_makes_T_surf_lower_than_T_interior():
    """UTBL correction must lower T_surf below T_interior."""
    b = 1e-7
    T_interior = 3500.0
    T_surf = float(_utbl_tsurf_jax(jnp.asarray(T_interior), jnp.asarray(b)))
    assert T_surf < T_interior


@pytest.mark.unit
def test_apply_surface_bc_grey_off_utbl():
    """With param_utbl=False the surface flux uses T_interior^4."""
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=200.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=8000.0,
        core_heat_capacity=800.0,
        tfac_core_avg=1.147,
        param_utbl=False,
        param_utbl_const=0.0,
    )
    T_basic = jnp.asarray([2000.0, 2500.0, 3000.0])
    heat_flux = jnp.zeros((3,))
    out = _apply_surface_bc(heat_flux, bc, T_basic)

    SIGMA_SB = 5.670374419e-8  # CODATA, matches solver SIGMA_SB
    T_surf = 3000.0  # T_basic[-1], no UTBL correction
    expected = SIGMA_SB * (T_surf**4 - 200.0**4)
    assert float(out[-1]) == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_apply_surface_bc_grey_on_utbl():
    """With param_utbl=True the surface flux uses T_surf^4 < T_interior^4."""
    b = 1e-7
    T_interior = 3000.0
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=200.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=8000.0,
        core_heat_capacity=800.0,
        tfac_core_avg=1.147,
        param_utbl=True,
        param_utbl_const=b,
    )
    T_basic = jnp.asarray([2000.0, 2500.0, T_interior])
    heat_flux = jnp.zeros((3,))
    out = _apply_surface_bc(heat_flux, bc, T_basic)

    SIGMA_SB = 5.670374419e-8
    T_surf = float(_utbl_tsurf_jax(jnp.asarray(T_interior), jnp.asarray(b)))
    expected = SIGMA_SB * (T_surf**4 - 200.0**4)
    assert float(out[-1]) == pytest.approx(expected, rel=1e-10)
    # And the UTBL flux MUST be lower than the no-UTBL flux at the same T.
    out_no_utbl_bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=200.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=8000.0,
        core_heat_capacity=800.0,
        tfac_core_avg=1.147,
        param_utbl=False,
        param_utbl_const=0.0,
    )
    out_no_utbl = _apply_surface_bc(heat_flux, out_no_utbl_bc, T_basic)
    assert float(out[-1]) < float(out_no_utbl[-1])
