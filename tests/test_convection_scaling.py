"""Nusselt-Rayleigh scaling of the mixing-length convective closure.

Module under test: ``aragog.solver.entropy_state.EntropyState`` (the
per-timestep convective flux engine of ``EntropySolver``), driven in
constant-property mode so the mixing-length closure is isolated from the
EOS tables.

The convective heat flux the solver transports is the mixing-length-theory
(MLT) closure of Abe (1995) as implemented by Bower et al. (2018) section
2.1: an eddy diffusivity that blends a viscous (laminar) branch and an
inviscid (free-fall / soft-turbulence) branch across the critical Reynolds
number ``Re_crit = 9/8``. In terms of the superadiabatic entropy gradient
``s = -dS/dr`` and the dynamic viscosity ``eta``:

- viscous branch:  ``kappa_h ~ l^4 s / (18 nu)``  ->  ``F_conv ~ s^2 / nu``
- inviscid branch: ``kappa_h ~ l^2 (alpha g T s / c_p)^{1/2}``  ->  ``F_conv ~ s^{3/2}``

The inviscid free-fall flux ``F_conv ~ s^{3/2}`` is the high-Rayleigh-number
limit that boundary-layer theory closes with the soft-turbulence Nusselt-
Rayleigh law ``Nu ~ Ra^{1/3}`` (Solomatov 2007); aragog recovers that law
as a diagnostic rather than imposing it (see docs/Explanations/mixing_length.md).
This file pins the underlying closure exponents directly, because those are
the exact, deterministic outputs of the flux engine and a regression in the
free-fall power (1.5 -> 1.0 or 2.0) is exactly the failure a Nu-Ra fit would
diagnose downstream.

Invariants exercised: positivity (no convective flux across a stable
gradient), the free-fall and viscous power laws (reference-pinned against
Abe 1995 / Bower 2018), and the Reynolds-number regime separation.

References: Abe (1995); Bower, Sanan & Wolf (2018),
https://doi.org/10.1016/j.pepi.2017.11.004, section 2.1; Solomatov (2007).
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import RE_CRIT, EntropyState

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

# Layer geometry and constant material properties. A deep mantle shell so the
# mixing length (distance to nearest boundary) spans ~1e6 m in the interior.
_R_CMB = 3.480e6
_R_SURF = 6.371e6
_G = 9.81
_RHO = 4000.0
_CP = 1000.0
_ALPHA = 3.0e-5
_K = 4.0
_T_REF = 2000.0
_S_REF = 3000.0

# Interior sampling window: exclude the three nodes nearest each wall where the
# nearest-boundary mixing length collapses to zero and the closure is by
# construction conduction-only.
_INTERIOR = slice(3, -3)


def _make_mesh(n=60):
    """Radial mesh with the SPIDER-standard nearest-boundary mixing length."""
    r_stag = np.linspace(_R_CMB, _R_SURF, n)
    dr = np.diff(r_stag)
    r_basic = np.zeros(n + 1)
    r_basic[0], r_basic[-1] = _R_CMB, _R_SURF
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    p_stag = np.linspace(135e9, 1e5, n)
    p_basic = np.interp(r_basic, r_stag, p_stag)

    class _Sub:
        pass

    class _Mesh:
        pass

    mesh = _Mesh()
    mesh.basic = _Sub()
    mesh.staggered = _Sub()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - _R_CMB, _R_SURF - r_basic), 1.0)
    mesh.basic.mixing_length = ml
    mesh.basic.mixing_length_squared = ml**2
    mesh.basic.mixing_length_cubed = ml**3
    mesh.basic.pressure = p_basic
    mesh.staggered.pressure = p_stag
    mesh.basic.mass_radii = r_basic
    mesh.staggered.mass_radii = r_stag
    mesh.dxidr = np.ones_like(r_basic)
    mesh.dr = dr
    mesh.N = n

    def q_basic(q):
        q = np.asarray(q).flatten()
        out = np.zeros(n + 1)
        out[0], out[-1] = q[0], q[-1]
        out[1:-1] = 0.5 * (q[:-1] + q[1:])
        return out

    def d_dr_basic(q):
        q = np.asarray(q).flatten()
        out = np.zeros(n + 1)
        out[1:-1] = np.diff(q) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    mesh.quantity_at_basic_nodes = q_basic
    mesh.d_dr_at_basic_nodes = d_dr_basic
    return mesh


def _make_state(mesh, log10visc):
    """Constant-property EntropyState with plastic yielding disabled, so the
    viscosity is a pure Newtonian constant and the closure is isoviscous."""

    def _evaluator(pressure):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=_G,
            const_properties=True,
            const_rho=_RHO,
            const_Cp=_CP,
            const_alpha=_ALPHA,
            const_cond=_K,
            const_log10visc=log10visc,
            const_T_ref=_T_REF,
            const_S_ref=_S_REF,
            yield_stress_c=1.0e30,
            yield_stress_mu=0.0,
            yield_stress_max=1.0e40,
            stress_closure_mode='local',
        )
        ev.set_pressure(pressure)
        return ev

    class _Eval:
        pass

    evaluator = _Eval()
    evaluator.mesh = mesh
    return EntropyState(
        evaluator=evaluator,
        phase_staggered=_evaluator(mesh.staggered.pressure),
        phase_basic=_evaluator(mesh.basic.pressure),
        conduction=True,
        convection=True,
    )


def _probe(log10visc, grad):
    """Impose a uniform superadiabatic entropy gradient ``dS/dr = grad`` and
    return the peak interior eddy diffusivity and convective flux."""
    mesh = _make_mesh()
    state = _make_state(mesh, log10visc)
    rs = np.asarray(mesh.staggered.radii).ravel()
    entropy = _S_REF + grad * (rs - rs.mean())
    state.update(entropy, 0.0)
    kappa_h = np.abs(np.asarray(state.eddy_diffusivity).ravel()[_INTERIOR])
    jconv = np.abs(np.asarray(state.jconv).ravel()[_INTERIOR])
    return float(np.max(kappa_h)), float(np.max(jconv))


def _slope(x, y):
    """Log-log least-squares slope."""
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


# A geometric ladder of superadiabatic gradients spanning two decades, so the
# fitted power is resolved well beyond the 0.5-vs-1.0-vs-1.5-vs-2.0 spacing that
# separates the physical branches.
_GRADS = np.array([-1.0e-7, -3.0e-7, -1.0e-6, -3.0e-6, -1.0e-5])


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_inviscid_freefall_flux_scales_as_gradient_three_halves():
    """In the inviscid (high-Reynolds) limit the convective flux follows the
    free-fall law ``F_conv ~ (-dS/dr)^{3/2}`` and the eddy diffusivity
    ``kappa_h ~ (-dS/dr)^{1/2}``.

    This free-fall flux is the high-Ra limit that boundary-layer theory
    closes as the soft-turbulence ``Nu ~ Ra^{1/3}`` law (Solomatov 2007), so
    the 3/2 power is the aragog-side content of that Nusselt-Rayleigh scaling.
    """
    kappa = []
    flux = []
    for g in _GRADS:
        kh, jc = _probe(1.0, g)  # log10visc = 1 -> molten, deep inviscid regime
        kappa.append(kh)
        flux.append(jc)

    beta_flux = _slope(-_GRADS, flux)
    beta_kappa = _slope(-_GRADS, kappa)
    # Free-fall exponents. rtol=2e-2 covers the tanh Re-blend leakage at finite
    # gradient; the branches are spaced by 0.5, far outside this window.
    assert beta_flux == pytest.approx(1.5, abs=0.02)
    assert beta_kappa == pytest.approx(0.5, abs=0.02)
    # Discrimination guard: the viscous branch would give flux ~ s^2 and the
    # conductive-mistaken case s^1; both are >= 0.5 away from 1.5.
    assert abs(beta_flux - 2.0) > 0.4
    assert abs(beta_flux - 1.0) > 0.4
    # Positivity: an unstable (negative) gradient drives a strictly positive
    # convective flux and eddy diffusivity.
    assert min(flux) > 0.0
    assert min(kappa) > 0.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_viscous_branch_scales_as_gradient_squared_over_viscosity():
    """In the viscous (low-Reynolds) limit the flux follows the laminar law
    ``F_conv ~ (-dS/dr)^2`` and the eddy diffusivity ``kappa_h ~ 1/nu``.

    Pins the Stokes-drag branch of the Abe/Bower closure and its inverse-
    viscosity dependence, the lever that sets the Rayleigh number in a
    stiff (solid) mantle.
    """
    flux = []
    for g in _GRADS:
        _, jc = _probe(20.0, g)  # log10visc = 20 -> stiff, deep viscous regime
        flux.append(jc)
    beta_flux = _slope(-_GRADS, flux)
    # Laminar exponent 2.0; discriminate against the inviscid 1.5.
    assert beta_flux == pytest.approx(2.0, abs=0.02)
    assert abs(beta_flux - 1.5) > 0.4

    # Inverse-viscosity law at fixed gradient: kappa_h ~ eta^{-1}.
    viscs = np.array([1.0e18, 1.0e19, 1.0e20, 1.0e21])
    kappa = [_probe(np.log10(v), -1.0e-6)[0] for v in viscs]
    beta_nu = _slope(viscs, kappa)
    assert beta_nu == pytest.approx(-1.0, abs=0.02)
    # Sign/scale guard: doubling nowhere flips the sign; kappa strictly drops.
    assert kappa[0] > kappa[-1] > 0.0


@pytest.mark.physics_invariant
def test_reynolds_regime_separates_viscous_from_inviscid():
    """The eddy diffusivity is viscosity-independent in the inviscid limit and
    inversely proportional to viscosity in the viscous limit, with the two
    regimes straddling the critical Reynolds number ``Re_crit = 9/8``.

    Edge behaviour: a stable (positive) entropy gradient must transport no
    convective flux in either regime.
    """
    grad = -1.0e-6
    kh_molten_1 = _probe(1.0, grad)[0]
    kh_molten_2 = _probe(2.0, grad)[0]
    # Inviscid: a decade of viscosity change leaves kappa_h essentially fixed.
    assert kh_molten_2 == pytest.approx(kh_molten_1, rel=1e-3)

    kh_stiff_1 = _probe(20.0, grad)[0]
    kh_stiff_2 = _probe(21.0, grad)[0]
    # Viscous: a decade of viscosity change moves kappa_h by ~10x.
    assert kh_stiff_1 / kh_stiff_2 == pytest.approx(10.0, rel=0.05)

    # The critical Reynolds number is the documented Abe (1995) value.
    assert RE_CRIT == pytest.approx(9.0 / 8.0, rel=1e-12)

    # Edge case: a stable stratification (dS/dr > 0) shuts convection off.
    _, jc_stable = _probe(1.0, +1.0e-6)
    assert jc_stable == pytest.approx(0.0, abs=1e-6)
    # ... while the mirror-image unstable gradient convects vigorously, so the
    # near-zero above is a real shutdown, not a dead probe.
    _, jc_unstable = _probe(1.0, -1.0e-6)
    assert jc_unstable > 1.0

@pytest.mark.physics_invariant
@pytest.mark.slow

@pytest.mark.physics_invariant
@pytest.mark.slow
def test_jax_compute_mlt_convection_scaling():
    """Verify that the JAX MLT kernel obeys the inviscid (free-fall) and viscous Nu-Ra scaling exponents."""
    pytest.importorskip('jax')
    import jax.numpy as jnp
    from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt
    from aragog.solver.entropy_state import RE_CRIT
    
    n_basic = 20
    n = n_basic - 1
    # Create simple mesh with mixing length = 1.0e5
    r_basic = jnp.linspace(3.48e6, 6.371e6, n_basic)
    r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
    mesh = MeshArrays(
        d_dr_matrix=jnp.zeros((n_basic, n)),
        quantity_matrix=jnp.zeros((n_basic, n)),
        area=jnp.ones(n_basic),
        volume=jnp.ones(n_basic),
        radii_basic=r_basic,
        radii_stag=r_stag,
        mixing_length=jnp.full(n_basic, 1.0e5),
        mixing_length_sq=jnp.full(n_basic, 1.0e10),
        mixing_length_cu=jnp.full(n_basic, 1.0e15),
        P_stag=jnp.zeros(n),
        P_basic=jnp.zeros(n_basic),
        dP_dr_basic=jnp.zeros(n_basic),
        gravity=jnp.full(n_basic, 9.81),
    )
    
    ones = jnp.ones(n_basic)
    phase = PhaseProperties(
        temperature=ones * 2000.0,
        density=ones * 4000.0,
        heat_capacity=ones * 1200.0,
        thermal_expansivity=ones * 3e-5,
        dTdPs=ones,
        melt_fraction=ones * 0.0,
        viscosity=ones * 1.0e21,
        kinematic_viscosity=ones * 1.0e21 / 4000.0,
        thermal_conductivity=ones * 4.0,
        latent_heat=ones,
        capacitance=ones * 4000.0 * 2000.0,
        eta_diff=ones * 1.0e21,
        tau_y=ones * 1.0e9, visc_solid_weight=ones * 1.0,
    )
    
    # Varied superadiabatic gradient
    ds_dr_array = -jnp.logspace(-8, -2, 10)
    
    # First, inviscid:
    phase_inv = phase._replace(
        viscosity=ones * 1.0e10, 
        kinematic_viscosity=ones * 1.0e10 / 4000.0
    )
    k_inv = []
    for ds_dr in ds_dr_array:
        k_h, _ = compute_mlt(jnp.full(n_basic, ds_dr), phase_inv, mesh, PhaseParams(kappah_floor=0.0))
        k_inv.append(float(k_h[10]))
    
    # In free-fall, F_conv ~ (-dS/dr)^{1.5}, so k_h ~ (-dS/dr)^{0.5}
    beta_inv = np.polyfit(np.log10(-ds_dr_array), np.log10(k_inv), 1)[0]
    assert beta_inv == pytest.approx(0.5, abs=0.05)
    
    # Now viscous:
    phase_visc = phase._replace(
        viscosity=ones * 1.0e21, 
        kinematic_viscosity=ones * 1.0e21 / 4000.0
    )
    k_visc = []
    for ds_dr in ds_dr_array:
        k_h, _ = compute_mlt(jnp.full(n_basic, ds_dr), phase_visc, mesh, PhaseParams(kappah_floor=0.0))
        k_visc.append(float(k_h[10]))
    
    # In viscous, F_conv ~ (-dS/dr)^{2}, so k_h ~ (-dS/dr)^{1.0}
    beta_visc = np.polyfit(np.log10(-ds_dr_array), np.log10(k_visc), 1)[0]
    assert beta_visc == pytest.approx(1.0, abs=0.05)
