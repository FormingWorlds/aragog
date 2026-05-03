"""Energy source configuration."""

from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class EnergyConfig:
    """Physics toggle flags and heating parameters.

    Parameters
    ----------
    conduction : bool
    convection : bool
    gravitational_separation : bool
    mixing : bool
    radionuclides : bool
    tidal : bool
    eddy_diffusivity_chemical : float
        Ratio kappa_c / kappa_h for chemical eddy diffusivity.
    tidal_array : ndarray
        Tidal heating per unit mass [W/kg] at each layer.
    """

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool
    eddy_diffusivity_chemical: float = 1.0
    tidal_array: npt.NDArray = attrs.Factory(lambda: np.array([0.0], dtype=float))
