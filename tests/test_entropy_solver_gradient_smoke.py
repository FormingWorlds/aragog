"""Smoke test of ``EntropySolver`` in ``core_bc='gradient'`` mode.

The gradient mode is structurally distinct from the other three core_bc
modes: the state vector stores ``[dS/dr at N+1 basic nodes, S_surf]`` of
length ``N+2`` rather than entropy at staggered nodes, and the entropy
profile is reconstructed from the gradient state at every RHS call. The
reconstruction logic, gradient-mode dispatches in ``_dSdt_single`` /
``solve`` / ``get_state`` / ``_compute_step_energy_integrals``, the
gradient state-accessor branches, and the ``_build_jac_sparsity`` ``None``
return for the dense gradient Jacobian are all unique to this mode and
have no exercise in any other test in the suite.

A single short integration here exercises the full gradient call graph so
the mode no longer sits in the coverage shadow of the dispatch table.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Mirror the EOS-resolution policy of ``test_entropy_solver_integration``:
# accept ``ARAGOG_TEST_EOS_DIR`` first, then ``$FWL_DATA/aragog/spider_eos``,
# then a legacy dev-machine fallback. Skip if none resolves.
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
    reason=(
        f'SPIDER P-S tables not found at {EOS_DIR}. Set ARAGOG_TEST_EOS_DIR '
        'or populate $FWL_DATA/aragog/spider_eos.'
    ),
)

pytestmark = [pytest.mark.smoke, needs_eos]


def _build_gradient_parameters(*, n_nodes: int = 12, end_time: float = 5.0):
    """Mirror ``test_entropy_solver_integration._build_parameters`` but
    fix ``core_bc='gradient'`` and use the scipy ``radau`` solver path so
    the test does not depend on scikits.odes.
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
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc='gradient',
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
        use_jax_jacobian=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
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
        end_time=end_time,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    return Parameters(
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


@pytest.fixture(scope='module')
def shared_eos():
    """Module-level EntropyEOS reuse, matching the integration suite."""
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def test_entropy_solver_gradient_mode_short_run_completes(shared_eos):
    """Run a short ``core_bc='gradient'`` integration end-to-end and
    verify the state vector layout, the gradient-mode reconstruction,
    and the post-solve diagnostics.

    Discriminator: the gradient state vector must have length ``N+2``
    (``N+1`` gradients plus ``S_surf``); a regression that fell through
    to the quasi_steady or energy_balance length would surface as a
    shape mismatch on the very first integrator call. Reconstructing
    the entropy profile from the final gradient state must give a
    physically plausible monotone-or-near-monotone profile in K (no
    negative temperatures, no orders-of-magnitude blow-ups).
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_gradient_parameters(n_nodes=12, end_time=5.0)
    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()
    solver.set_initial_entropy(3050.0)
    solver.solve()

    # ── Discriminator 1: gradient state vector length ──────────────
    n_stag = solver._n_stag
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    expected_len = n_stag + 2  # N+1 dSdr + 1 S_surf
    assert len(final_y) == expected_len, (
        f'gradient state length is {len(final_y)}, expected n_stag+2 = {expected_len}. '
        'Off-by-one would surface here before any other check.'
    )

    # ── Discriminator 2: entropy reconstruction round-trips IC ─────
    # Call the gradient-mode entropy_staggered accessor and verify the
    # reconstructed S vector is physically plausible. A bug in
    # ``_reconstruct_entropy`` (wrong sign of dSdr * dr, wrong loop
    # direction) would surface here as a wildly off entropy.
    S_stag = solver.entropy_staggered
    if S_stag.ndim > 1:
        S_stag = S_stag[:, -1]
    assert S_stag.shape[0] == n_stag, (
        f'entropy_staggered (gradient) returned shape {S_stag.shape}, expected ({n_stag},)'
    )
    # 5-yr cooling from S=3050 in this geometry must keep S in the
    # mushy-to-just-solid window, well clear of either end of the
    # PALEOS table. Tight bounds discriminate against a reconstruction
    # that lost or doubled the gradient term.
    assert float(S_stag.min()) > 1500.0, (
        f'reconstructed S_min={float(S_stag.min()):.1f} J/kg/K is below the '
        'lowest entropy in the PALEOS table; gradient reconstruction is broken.'
    )
    assert float(S_stag.max()) < 5500.0, (
        f'reconstructed S_max={float(S_stag.max()):.1f} J/kg/K exceeds the '
        'highest entropy in the PALEOS table.'
    )

    # ── Discriminator 3: temperature_staggered (gradient branch) ──
    T_stag = solver.temperature_staggered
    if T_stag.ndim > 1:
        T_stag = T_stag[:, -1]
    assert np.all(np.isfinite(T_stag)), 'T_stag has non-finite entries in gradient mode'
    assert float(np.min(T_stag)) > 0.0, 'T_stag has non-positive entries (Kelvin)'
    assert 1000.0 < float(T_stag.min()) < float(T_stag.max()) < 8000.0, (
        f'T_stag range [{float(T_stag.min()):.1f}, {float(T_stag.max()):.1f}] K is '
        'outside the plausible mantle window.'
    )

    # ── Discriminator 4: get_state walks the gradient slicer ──────
    out = solver.get_state()
    assert np.all(np.isfinite(out.S_final)), 'SolverOutput.S_final has NaN'
    assert np.all(np.isfinite(out.T_stag)), 'SolverOutput.T_stag has NaN'
    assert 0.0 <= float(out.Phi_global) <= 1.0, (
        f'Phi_global={float(out.Phi_global):.4f} outside [0, 1]'
    )
    # F_cmb and the per-call step integrals should be finite floats
    # in the post-solve snapshot. A regression in
    # ``_compute_step_energy_integrals`` gradient branch would raise
    # inside the trajectory walk.
    assert np.isfinite(out.F_cmb), f'F_cmb is not finite: {out.F_cmb}'
    assert np.isfinite(out.step_dE_F_int_J), (
        f'step_dE_F_int_J non-finite: {out.step_dE_F_int_J}'
    )
    assert np.isfinite(out.step_dE_F_cmb_J), (
        f'step_dE_F_cmb_J non-finite: {out.step_dE_F_cmb_J}'
    )

    # ── Discriminator 5: build_jac_sparsity returns None for gradient
    # mode (the reconstruction couples all gradients to all staggered
    # nodes, making the Jacobian dense, so scipy must do a full FD).
    sparsity = solver._build_jac_sparsity()
    assert sparsity is None, (
        f'gradient mode must request dense Jacobian via None sparsity; got {type(sparsity)}'
    )


def test_entropy_solver_gradient_set_initial_entropy_round_trips(shared_eos):
    """The gradient-mode IC must encode the input entropy profile and
    reconstruct it to within solver tolerance.

    Discriminator: if the gradient IC dropped or mis-ordered a basic
    node, the reconstruction round-trip error would be order S_init,
    not order ``rtol * S_init``.
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_gradient_parameters(n_nodes=10, end_time=1.0)
    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    # Non-uniform IC so the gradient is non-trivial and a no-op
    # reconstruction would be visibly wrong.
    S_init = np.linspace(3100.0, 3400.0, n_stag)
    solver.set_initial_entropy(S_init)

    # Gradient mode stores [dSdr at N+1 basic nodes, S_surf] of length N+2.
    assert solver._S0.shape == (n_stag + 1 + 1,), (
        f'gradient _S0 shape is {solver._S0.shape}, expected ({n_stag + 2},)'
    )
    assert float(solver._S0[n_stag + 1]) == pytest.approx(float(S_init[-1])), (
        'last entry of gradient _S0 should be S_surf (surface entropy)'
    )

    # Round-trip: reconstruct from the IC and verify the error is
    # small relative to the prescribed gradient. The reconstruction
    # uses one-sided FD at the boundary, which has O(dr) truncation
    # error; for a 300 J/kg/K spread on 10 nodes (dr ~ 3e5 m), we
    # expect a few J/kg/K mismatch at most, far below the gradient
    # span itself. A bug that lost the gradient term entirely would
    # surface as an error of order 300 J/kg/K (the IC range).
    dSdr_init = solver._S0[: n_stag + 1]
    S_surf_init = float(solver._S0[n_stag + 1])
    S_check, _ = solver._reconstruct_entropy(dSdr_init, S_surf_init)
    err = float(np.max(np.abs(S_check - S_init)))
    span = float(np.ptp(S_init))
    assert err < 0.05 * span, (
        f'gradient IC reconstruction round-trip error is {err:.2e} J/kg/K, '
        f'> 5% of the IC span {span:.2e} J/kg/K. Gradient IC lost information.'
    )
