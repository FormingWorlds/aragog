"""Standalone diagnostic functions for Aragog output analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aragog.eos import EntropyEOS
from aragog.mesh import Mesh
from aragog.utilities import FloatOrArray


def total_enthalpy(
    eos: EntropyEOS,
    P_stag: npt.NDArray,
    S_stag: npt.NDArray,
    mass_stag: npt.NDArray,
) -> float:
    """Mass-integrated specific enthalpy of the mantle [J].

    Computed via the EOS-consistent ``h(P, S)`` table that
    ``EntropyEOS`` precomputes from the fundamental thermodynamic
    relation ``dh = T dS + (1/rho) dP``. Latent heat is captured
    automatically because the integration path crosses the mushy
    zone at constant temperature while entropy sweeps across the
    phase transition.

    Parameters
    ----------
    eos : EntropyEOS
        EOS object whose ``specific_enthalpy(P, S)`` lookup table has
        already been built (done in ``EntropyEOS.__init__``).
    P_stag : ndarray
        Pressure at staggered nodes [Pa].
    S_stag : ndarray
        Entropy at staggered nodes [J/kg/K].
    mass_stag : ndarray
        Mass per shell at staggered nodes [kg].

    Returns
    -------
    float
        Total enthalpy [J]. The absolute value is anchor-dependent
        (zero is fixed by the EOS table corner) but the additive
        constant cancels in any time difference, which is what the
        conservation diagnostic uses.
    """
    h_stag = np.asarray(eos.specific_enthalpy(P_stag, S_stag)).ravel()
    mass = np.asarray(mass_stag).ravel()
    return float(np.dot(h_stag, mass))


def volume_average(mesh: Mesh, staggered_quantity: npt.NDArray) -> float:
    """Compute the volume-weighted average of a quantity on the staggered mesh.

    Parameters
    ----------
    mesh : Mesh
        The staggered mesh.
    staggered_quantity : npt.NDArray
        A 1D array of values at staggered nodes.

    Returns
    -------
    float
        Volume-averaged value.
    """
    return np.dot(staggered_quantity.T, mesh.basic.volume).item() / mesh.basic.total_volume


def melt_fraction_global(
    mesh: Mesh,
    melt_fraction_staggered: FloatOrArray,
    phase_mode: str,
) -> float:
    """Compute the mass-weighted global melt fraction $M_\\mathrm{liq}/M_\\mathrm{mantle}$.

    Per-cell ``melt_fraction_staggered`` is the *mass* fraction of liquid in
    that cell (the entropy lever-rule fraction
    $(S - S_\\mathrm{sol})/(S_\\mathrm{liq} - S_\\mathrm{sol})$ equals the
    mass-fraction split between solid and melt by the lever rule, since
    entropy is per-unit-mass). The mass-weighted mantle average therefore
    gives $\\sum (m_i \\phi_i) / \\sum m_i = M_\\mathrm{liq}/M_\\mathrm{mantle}$,
    which is the quantity ``EntropySolver`` writes to the helpfile as
    ``Phi_global`` and the rootfn cap acts on. This function used to
    return a *volume-weighted* blend of per-cell mass-fractions, which is
    neither $M_\\mathrm{liq}/M_\\mathrm{mantle}$ nor $V_\\mathrm{liq}/V_\\mathrm{mantle}$
    and was inconsistent with the production helpfile path.

    For the porosity-derived volume-fraction $V_\\mathrm{liq}/V_\\mathrm{mantle}$,
    use ``SolverOutput.Phi_global_vol`` from a completed solve instead.

    Parameters
    ----------
    mesh : Mesh
        The staggered mesh. Must carry ``basic.volume`` (per-shell volume)
        and ``staggered_effective_density`` (per-shell structural density,
        set at IC and frozen for mass-coordinate consistency).
    melt_fraction_staggered : FloatOrArray
        Melt fraction (entropy lever-rule, i.e. local mass-fraction of
        liquid) on the staggered mesh. For time-resolved data this is
        typically a 2D array (nodes x times); only the last column is used.
    phase_mode : str
        Phase configuration string (e.g. "mixed", "composite", or a single-phase name).

    Returns
    -------
    float
        Mass-weighted mantle melt fraction $M_\\mathrm{liq}/M_\\mathrm{mantle} \\in [0, 1]$.
    """
    if phase_mode not in ('mixed', 'composite'):
        return melt_fraction_staggered

    phi = np.asarray(melt_fraction_staggered)
    if phi.ndim == 2:
        phi = phi[:, -1]
    phi = phi.ravel()
    volume = np.asarray(mesh.basic.volume).ravel()
    rho_struct = np.asarray(mesh.staggered_effective_density).ravel()
    mass = rho_struct * volume
    mass_total = float(mass.sum())
    if mass_total <= 0.0:
        return float(np.mean(phi))
    return float(np.dot(mass, phi) / mass_total)


def rheological_front(
    mesh: Mesh,
    melt_fraction_basic: npt.NDArray,
    rheological_transition_melt_fraction: float,
    phi_global: float,
    phi_high: float = 0.99,
    phi_low: float = 0.01,
) -> float:
    """Compute the dimensionless rheological front position.

    The rheological front is defined as the dimensionless depth (relative to
    the outer radius) where the melt fraction crosses the rheological
    transition threshold. The bypass thresholds ``phi_high`` and
    ``phi_low`` short-circuit the ``argmin`` search when the entire mantle
    is essentially liquid (no transition has formed yet) or essentially
    solid (transition has reached the surface). The defaults assume a
    bottom-up solidification mode; the search returns the *first* radial
    crossing of ``rheological_transition_melt_fraction`` from the CMB
    outward, so non-bottom-up modes (e.g. middle-out crystallisation)
    will need a redefined RF concept.

    Parameters
    ----------
    mesh : Mesh
        The staggered mesh.
    melt_fraction_basic : npt.NDArray
        Melt fraction on the basic mesh (nodes x times).
    rheological_transition_melt_fraction : float
        Melt fraction threshold for the rheological transition.
    phi_global : float
        Volume-averaged global melt fraction at the final timestep.
    phi_high : float, optional
        Upper bypass threshold (default 0.99). If ``phi_global`` is above
        this, RF is set to the inner radius (no transition yet).
    phi_low : float, optional
        Lower bypass threshold (default 0.01). If ``phi_global`` is below
        this, RF is set to the outer radius (fully solidified).

    Returns
    -------
    float
        Dimensionless rheological front (0 = surface, 1 = fully solidified to CMB).
    """
    # Bypass argmin when no transition is present in the mantle.
    if phi_global > phi_high:
        rf: float = mesh.basic.radii[0]
    elif phi_global < phi_low:
        rf = mesh.basic.radii[-1]
    # General case
    else:
        idx = np.argmin(
            np.abs(melt_fraction_basic[:, -1] - rheological_transition_melt_fraction)
        )
        rf = mesh.basic.radii[idx]

    # Return dimensionless rheological front
    return ((mesh.basic.radii[-1] - rf) / mesh.basic.radii[-1]).item()
