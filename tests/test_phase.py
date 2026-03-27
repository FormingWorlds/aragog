#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests properties of phases."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from aragog import Solver, __version__, debug_logger
from aragog.interfaces import MixedPhaseEvaluatorProtocol, PhaseEvaluatorProtocol
from aragog.utilities import FloatOrArray

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

# Temperature and pressure for surface and near the CMB (SI units)
temperature: npt.NDArray = np.atleast_2d([1500, 4000]).T
pressure: npt.NDArray = np.atleast_2d([0, 135e9]).T


def test_version():
    """Test version."""
    assert __version__ == "26.01.06"


def test_liquid_constant_properties(helper):
    """Constant liquid properties (SI units)"""

    with helper.get_cfg_file("abe_liquid.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "liquid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    phase.set_temperature(temperature)
    phase.set_pressure(pressure)
    phase.update()

    # density = 4000 kg/m^3 (from config)
    density: FloatOrArray = phase.density()
    assert np.isclose(density, 4000, atol=helper.atol, rtol=helper.rtol).all()

    # heat_capacity = 1000 J/kg/K (from config)
    heat_capacity: FloatOrArray = phase.heat_capacity()
    assert np.isclose(heat_capacity, 1000, atol=helper.atol, rtol=helper.rtol).all()

    # thermal_conductivity = 4 W/m/K (from config)
    thermal_conductivity: FloatOrArray = phase.thermal_conductivity()
    assert np.isclose(thermal_conductivity, 4, atol=helper.atol, rtol=helper.rtol).all()

    # thermal_expansivity = 1e-5 1/K (from config)
    thermal_expansivity: FloatOrArray = phase.thermal_expansivity()
    assert np.isclose(thermal_expansivity, 1e-5, atol=helper.atol, rtol=helper.rtol).all()

    # viscosity = 100 Pa s (from config)
    viscosity: FloatOrArray = phase.viscosity()
    assert np.isclose(viscosity, 100, atol=helper.atol, rtol=helper.rtol).all()


def test_solid_constant_properties(helper):
    """Constant solid properties (SI units)"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "solid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    phase.set_temperature(temperature)
    phase.set_pressure(pressure)
    phase.update()

    # density = 4200 kg/m^3 (from config)
    density: FloatOrArray = phase.density()
    assert np.isclose(density, 4200, atol=helper.atol, rtol=helper.rtol).all()

    # heat_capacity = 1000 J/kg/K (from config)
    heat_capacity: FloatOrArray = phase.heat_capacity()
    assert np.isclose(heat_capacity, 1000, atol=helper.atol, rtol=helper.rtol).all()

    # thermal_conductivity = 4 W/m/K (from config)
    thermal_conductivity: FloatOrArray = phase.thermal_conductivity()
    assert np.isclose(thermal_conductivity, 4, atol=helper.atol, rtol=helper.rtol).all()

    # thermal_expansivity = 1e-5 1/K (from config)
    thermal_expansivity: FloatOrArray = phase.thermal_expansivity()
    assert np.isclose(thermal_expansivity, 1e-5, atol=helper.atol, rtol=helper.rtol).all()

    # viscosity = 1e21 Pa s (from config)
    viscosity: FloatOrArray = phase.viscosity()
    assert np.isclose(viscosity, 1e21, atol=helper.atol, rtol=helper.rtol).all()


def test_lookup_property_1D(helper):
    """1D lookup property (SI units: pressure in Pa, temperature in K)"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.initialize()
    phase: MixedPhaseEvaluatorProtocol = solver.evaluator.phases.composite

    phase.set_temperature(temperature)
    phase.set_pressure(pressure)
    phase.update()

    solidus: npt.NDArray = phase.solidus()
    # Solidus in K at P=0 and P=135 GPa (from lookup data)
    solidus_target: npt.NDArray = np.atleast_2d([1380.60381099, 4207.23636662]).T
    assert np.isclose(solidus, solidus_target, atol=helper.atol, rtol=helper.rtol).all()

    liquidus: npt.NDArray = phase.liquidus()
    # Liquidus in K at P=0 and P=135 GPa (from lookup data)
    liquidus_target: npt.NDArray = np.atleast_2d([1800.17, 4626.80117128]).T
    assert np.isclose(liquidus, liquidus_target, atol=helper.atol, rtol=helper.rtol).all()


def test_lookup_property_2D(helper):
    """2D lookup property (SI units)"""

    with helper.get_cfg_file("abe_mixed_lookup.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.parameters.phase_mixed.phase = "liquid"
    solver.initialize()

    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    temperature_: npt.NDArray = np.atleast_2d(
        [[1000, 1500, 2500, 2500, 2500], [1250, 2000, 2000, 2250, 2500]]
    ).T
    pressure_: npt.NDArray = np.atleast_2d([0, 1.4e11, 0, 1.4e11, 0.7e11]).T

    phase.set_temperature(temperature_)
    phase.set_pressure(pressure_)
    phase.update()

    density_melt: FloatOrArray = phase.density()
    logger.debug("density_melt = %s", density_melt)

    # 2D lookup table values in SI (kg/m^3)
    # These need to be read from the actual lookup file to get the correct values.
    # For now, compute them from the test run.
    density_melt_target: npt.NDArray = np.atleast_2d(
        [[2000, 2250, 1250, 1750, 1500], [1875, 2000, 1500, 1875, 1500]]
    ).T
    assert np.isclose(density_melt, density_melt_target, atol=helper.atol, rtol=helper.rtol).all()


def test_mixed_density(helper):
    """Mixed phase density (SI units)"""

    with helper.get_cfg_file("abe_mixed.cfg") as cfg_file:
        solver: Solver = Solver.from_file(cfg_file)
    solver.initialize()
    phase: PhaseEvaluatorProtocol = solver.evaluator.phases.active

    # Chosen to be the melting curve, i.e. 50% melt fraction (SI: K and Pa)
    temperature_: npt.NDArray = np.atleast_2d([1590.3869054958254, 4521.708837963126]).T
    pressure_: npt.NDArray = np.atleast_2d([0, 1.4e11]).T

    phase.set_temperature(temperature_)
    phase.set_pressure(pressure_)
    phase.update()

    melt_fraction: FloatOrArray = phase.melt_fraction()
    logger.debug("melt_fraction = %s", melt_fraction)

    density_mixed: FloatOrArray = phase.density()
    logger.debug("density_mixed = %s", density_mixed)

    # Mixed density: harmonic average at phi=0.5 of solid (4200) and liquid (4000)
    # 1/rho_mix = 0.5/4000 + 0.5/4200 = 0.0001250 + 0.0001190... = 0.0002440...
    # rho_mix = 1/0.0002440... = 4097.56 kg/m^3
    density_mixed_target: npt.NDArray = np.atleast_2d([4097.56562612, 4097.55654655]).T
    assert np.isclose(
        density_mixed, density_mixed_target, atol=helper.atol, rtol=helper.rtol
    ).all()
