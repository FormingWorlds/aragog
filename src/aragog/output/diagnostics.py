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
    """Compute the volume-averaged global melt fraction.

    Parameters
    ----------
    mesh : Mesh
        The staggered mesh.
    melt_fraction_staggered : FloatOrArray
        Melt fraction on the staggered mesh. For time-resolved data this is
        typically a 2D array (nodes x times); only the last column is used.
    phase_mode : str
        Phase configuration string (e.g. "mixed", "composite", or a single-phase name).

    Returns
    -------
    float
        Global melt fraction.
    """
    if phase_mode == 'mixed' or phase_mode == 'composite':
        return volume_average(mesh, melt_fraction_staggered[:, -1])
    else:
        return melt_fraction_staggered


def rheological_front(
    mesh: Mesh,
    melt_fraction_basic: npt.NDArray,
    rheological_transition_melt_fraction: float,
    phi_global: float,
) -> float:
    """Compute the dimensionless rheological front position.

    The rheological front is defined as the dimensionless depth (relative to
    the outer radius) where the melt fraction crosses the rheological
    transition threshold.

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

    Returns
    -------
    float
        Dimensionless rheological front (0 = surface, 1 = fully solidified to CMB).
    """
    # If global melt fraction is close to one everywhere (magma ocean), rf is the inner radius
    if phi_global > 0.99:
        rf: float = mesh.basic.radii[0]
    # If global melt fraction is close to zero everywhere (solidified), rf is the outer radius
    elif phi_global < 0.01:
        rf = mesh.basic.radii[-1]
    # General case
    else:
        idx = np.argmin(
            np.abs(melt_fraction_basic[:, -1] - rheological_transition_melt_fraction)
        )
        rf = mesh.basic.radii[idx]

    # Return dimensionless rheological front
    return ((mesh.basic.radii[-1] - rf) / mesh.basic.radii[-1]).item()
