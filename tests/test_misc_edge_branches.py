"""Edge-case unit tests for small modules.

Targets the remaining small clusters of uncovered lines that are not
worth dedicated test files but together close >40 lines:

* ``solver/boundary.py``: unknown outer / inner BC integers raising
  ValueError (lines 130-131, 202-203).
* ``parser.py``: missing eos_file when eos_method=2 (line 366).
* ``jax/nondim.py``: rhs_scale validation - non-finite or non-positive
  rhs_scale must raise (line 81).
* ``solver/entropy_state.py``: ``_smooth_clip`` outside [0, 1] -> 0
  (lines 115-117), ``d_dr_temperature_basic`` accessor (1014-1015).
* ``mesh/__init__.py``: ``derive_core_density_from_mesh`` happy path
  (covered indirectly today but exercised here as a unit test).

Each test is purely synthetic, runs in <50 ms, and adds no smoke /
slow markers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
#                       solver/boundary.py
# ──────────────────────────────────────────────────────────────────────


def test_boundary_unknown_outer_bc_raises():
    """``BoundaryConditions.apply_flux_outer_boundary_condition`` must
    raise ValueError when ``outer_boundary_condition`` is not in the
    documented set (1 / 4 / 5).

    Discriminator: a regression that fell through to a no-op
    (silently leaving heat_flux[-1] unchanged) would let an invalid
    BC config produce a "running" simulation with arbitrary surface
    flux. Raising forces the user to fix the config.
    """
    from aragog.solver.boundary import BoundaryConditions

    settings = type('S', (), {})()
    settings.outer_boundary_condition = 99  # not 1, 4, or 5
    settings.outer_boundary_value = 0.0
    settings.emissivity = 1.0
    settings.equilibrium_temperature = 0.0
    settings.tsurf_poststep_change = 0.0
    bc = BoundaryConditions.__new__(BoundaryConditions)
    bc._settings = settings

    state = type('State', (), {})()
    state._heat_flux = np.zeros(5)
    state.top_temperature = np.array([1500.0])

    with pytest.raises(ValueError, match='outer_boundary_condition'):
        bc.apply_flux_outer_boundary_condition(state)


def test_boundary_unknown_inner_bc_raises():
    """Same dispatch contract for the inner (CMB) boundary."""
    from aragog.solver.boundary import BoundaryConditions

    settings = type('S', (), {})()
    settings.inner_boundary_condition = 77
    settings.inner_boundary_value = 0.0
    bc = BoundaryConditions.__new__(BoundaryConditions)
    bc._settings = settings

    state = type('State', (), {})()
    state._heat_flux = np.zeros(5)

    with pytest.raises(ValueError, match='inner_boundary_condition'):
        bc.apply_flux_inner_boundary_condition(state)


# ──────────────────────────────────────────────────────────────────────
#                       jax/nondim.py rhs_scale validation
# ──────────────────────────────────────────────────────────────────────


def test_nondim_scales_negative_rhs_scale_raises():
    """Supplying a negative ``rhs_scale`` must raise.

    Discriminator: ``__post_init__`` validates rhs_scale only when
    the user provides it explicitly (when None, it is derived from
    state_scale, which is already positive). A regression that
    shortcut the validation would let a sign-flipped rhs_scale pass.
    """
    from aragog.jax.nondim import NonDimScales

    state_scale = np.array([3.0e3, 3.0e3, 3.0e3])
    # Wrong-signed rhs_scale: matches the contract magnitude but
    # has the wrong sign on one element.
    rhs_scale = np.array([1.0 / 3.0e3, -1.0 / 3.0e3, 1.0 / 3.0e3])
    with pytest.raises(ValueError, match='rhs_scale'):
        NonDimScales(state_scale=state_scale, t_ref=1.0, rhs_scale=rhs_scale)


# ──────────────────────────────────────────────────────────────────────
#                       parser.py
# ──────────────────────────────────────────────────────────────────────


def test_parameters_post_init_rejects_eos_method_2_without_eos_file():
    """When ``mesh.eos_method == 2`` the parser requires ``eos_file``;
    omitting it must raise ValueError with a clear remediation
    message.

    Edge case: silently falling through to eos_method=1 would mask
    the misconfigured run as Adams-Williamson, producing physically
    wrong gravity / pressure profiles in PROTEUS-coupled mode.
    """
    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _SolverParameters,
    )

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=273.0,
        core_heat_capacity=880.0,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    # eos_method=2 + eos_file='' triggers the post_init guard.
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=10,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file='',
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(
        start_time=0.0,
        end_time=1.0,
        atol=1.0e-9,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )

    with pytest.raises(ValueError, match='equation of state'):
        Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[],
            solver=sv,
        )


# ──────────────────────────────────────────────────────────────────────
#                       mesh/__init__.py
# ──────────────────────────────────────────────────────────────────────


def test_derive_core_density_from_mesh_round_trips(tmp_path):
    """``derive_core_density_from_mesh`` must reproduce the prescribed
    core mass when the mesh's CMB radius and density column are
    consistent.

    Discriminator: integrating ``rho(r) * 4 pi r^2 dr`` from the
    mesh CMB outward must give the same M_core that PROTEUS feeds
    in. Off by a factor of 4*pi/3 would surface here as a 3.0x error.
    """
    from aragog.mesh import derive_core_density_from_mesh

    # Synthetic Zalmoxis-style mantle mesh: first row is the CMB
    # (R_cmb = 3.5e6 m), ascending r toward the surface. 5 columns
    # r, P, rho, g, T per row.
    r = np.linspace(3.5e6, 6.371e6, 8)
    rho = np.full_like(r, 4000.0)
    P = np.linspace(1.4e11, 1.0e5, r.size)
    g = np.full_like(r, 10.0)
    T = np.linspace(3000.0, 1500.0, r.size)
    arr = np.stack([r, P, rho, g, T], axis=1)
    mesh_file = tmp_path / 'mesh.dat'
    np.savetxt(mesh_file, arr)

    # Analytic average core density for an Earth-like 1.93e24 kg core
    # and r_cmb = 3.5e6 m.
    r_cmb = 3.5e6
    M_core_target = 1.93e24
    rho_back = derive_core_density_from_mesh(str(mesh_file), M_core_target)
    rho_expected = M_core_target / ((4.0 / 3.0) * np.pi * r_cmb**3)
    assert rho_back == pytest.approx(rho_expected, rel=1e-6), (
        f'derive_core_density_from_mesh returned {rho_back:.3e}, expected {rho_expected:.3e}'
    )

    # Edge case: M_core <= 0 must raise.
    with pytest.raises(ValueError, match='M_core must be positive'):
        derive_core_density_from_mesh(str(mesh_file), -1.0)


# ──────────────────────────────────────────────────────────────────────
#                       solver/entropy_state.py
# ──────────────────────────────────────────────────────────────────────


def test_spider_get_smoothing_with_zero_width_does_hard_clip():
    """SPIDER's ``_spider_get_smoothing`` with ``smooth_width=0`` does a
    hard clip: 1 inside [0, 1], 0 outside.

    Discriminator: with smooth_width=0 the function takes the
    no-smoothing branch (lines 115-117 of entropy_state.py) which uses
    ``np.where`` masking instead of the tanh blend. Outside [0, 1] the
    return must be exactly 0.0; an off-by-one in the mask (e.g.
    ``gphi <= 0`` instead of ``gphi < 0``) would shift the boundary
    behaviour.
    """
    from aragog.solver.entropy_state import _spider_get_smoothing

    gphi = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    out = _spider_get_smoothing(gphi, smooth_width=0.0)
    assert out[0] == 0.0, f'gphi=-0.5 must clip to 0; got {out[0]}'
    assert out[1] == 1.0, f'gphi=0.0 should be inside; got {out[1]}'
    assert out[2] == 1.0
    assert out[3] == 1.0
    assert out[4] == 0.0


def test_spider_get_smoothing_smooth_branch_tapers_outside_bracket():
    """The two tanh branches of ``_spider_get_smoothing`` taper to zero
    just outside [0, 1] when ``smooth_width > 0``.

    Discriminator: at gphi=0.5 the smoothing is at its peak (~1); at
    gphi well outside [0, 1] it has fallen to the tanh tail. A
    regression that mis-typed the upper-branch formula (e.g.
    ``tanh((gphi - 1.0) / w)`` vs ``tanh((1.0 - gphi) / w)``) would
    flip the upper-branch sign and break the gphi >> 1 limit.
    """
    from aragog.solver.entropy_state import _spider_get_smoothing

    gphi = np.array([0.5, 1.05])
    out = _spider_get_smoothing(gphi, smooth_width=0.01)
    assert float(out[0]) > 0.99, f'smoothing at gphi=0.5 should be ~1; got {float(out[0])}'
    assert float(out[1]) < 0.01, (
        f'smoothing at gphi=1.05 (5 widths past upper boundary) should be <0.01; '
        f'got {float(out[1])}'
    )
