"""Entropy-formulation solver for Aragog.

Solves the entropy equation:
    rho * T * dS/dt = (1/r^2) d/dr [r^2 F_total] + rho * H

where S is specific entropy [J/kg/K], F_total is the total energy flux
[W/m^2], and H is the internal heating rate [W/kg].

This replaces the temperature-based Solver for PALEOS EOS runs.
Uses the same BDF time integrator (scipy solve_ivp) and staggered
finite-volume mesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult
from scipy.sparse import diags as sparse_diags

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters
from aragog.solver.boundary import BoundaryConditions
from aragog.solver.entropy_state import EntropyState

# Import SECS_PER_YEAR directly to avoid circular import with solver/__init__.py
from scipy import constants as _sp_constants
SECS_PER_YEAR: float = _sp_constants.Julian_year

logger = logging.getLogger(__name__)


@dataclass
class SolverOutput:
    """Complete output from one EntropySolver integration step.

    This dataclass is the public contract between Aragog and PROTEUS.
    All quantities needed by the coupling wrapper are included here,
    so callers never need to reach into solver internals.
    """

    # Profiles at staggered nodes
    S_final: npt.NDArray        # entropy [J/kg/K]
    T_stag: npt.NDArray         # temperature [K]
    phi_stag: npt.NDArray       # melt fraction [-]
    rho_stag: npt.NDArray       # density [kg/m^3]
    visc_stag: npt.NDArray      # dynamic viscosity [Pa s]

    # Mesh geometry
    P_stag: npt.NDArray         # pressure at staggered nodes [Pa]
    r_basic: npt.NDArray        # radii at basic nodes [m]
    r_stag: npt.NDArray         # radii at staggered nodes [m]
    vol: npt.NDArray            # shell volumes [m^3]
    mass_stag: npt.NDArray      # mass per shell [kg]

    # Fluxes and heating (at basic / staggered nodes)
    heat_flux: npt.NDArray      # total heat flux at basic nodes [W/m^2]
    heating: npt.NDArray        # internal heating at staggered nodes [W/kg]
    eddy_diff: npt.NDArray      # eddy diffusivity at basic nodes [m^2/s]
    cap_stag: npt.NDArray       # capacitance rho*T at staggered nodes

    # Scalar quantities
    T_magma: float              # surface temperature [K]
    T_core: float               # CMB temperature [K]
    Phi_global: float           # volume-weighted melt fraction
    Phi_global_vol: float       # porosity-based volumetric melt fraction
    M_mantle: float             # mantle mass [kg]
    M_mantle_liquid: float      # liquid mantle mass [kg]
    M_mantle_solid: float       # solid mantle mass [kg]
    RF_depth: float             # rheological front depth (dimensionless)
    E_th: float                 # thermal energy [J]
    Cp_eff: float               # effective heat capacity [J/kg/K]
    F_heat_total: float         # total heating flux [W/m^2]
    dt_actual: float            # actual integration time [yr]
    status: int                 # solver status (0 = success)


class EntropySolver:
    """Entropy-based interior dynamics solver.

    Drop-in replacement for Solver (T-based) when using PALEOS P-S tables.
    Same interface: initialize() -> set_initial_entropy() -> solve().
    PROTEUS can swap Solver for EntropySolver without changing the wrapper.

    Parameters
    ----------
    parameters : Parameters
        Parsed configuration (same as T-based Solver).
    entropy_eos : EntropyEOS
        Loaded P-S EOS tables.
    """

    def __init__(self, parameters: Parameters, entropy_eos: EntropyEOS):
        self.parameters = parameters
        self.entropy_eos = entropy_eos
        self.evaluator: object
        self.state: EntropyState
        self._solution: OptimizeResult
        self.stop_early: bool = False

    @classmethod
    def from_file(cls, filename: str, eos_dir: str, root: str = '') -> 'EntropySolver':
        """Create EntropySolver from a config file and EOS directory.

        Parameters
        ----------
        filename : str
            Path to TOML configuration file.
        eos_dir : str
            Path to directory with SPIDER-format P-S tables.
        root : str
            Root directory for the config file.
        """
        config_path = Path(root) / Path(filename)
        parameters = Parameters.from_file(config_path)
        entropy_eos = EntropyEOS(Path(eos_dir))
        return cls(parameters, entropy_eos)

    def initialize(self) -> None:
        """Initialize mesh, boundary conditions, and entropy state.

        Unlike the T-based Solver, we only need the mesh and BCs from
        the Evaluator. The T-based phase evaluators (which require
        solidus/liquidus files) are replaced by EntropyPhaseEvaluator.
        """
        logger.info('Initializing EntropySolver')
        self._initialize_internals()

    def _initialize_internals(self) -> None:
        """Build mesh, BCs, and entropy phase evaluators."""

        # Build mesh and BCs without the T-based phases
        from aragog.mesh import Mesh
        mesh = Mesh(self.parameters)
        bc = BoundaryConditions(self.parameters, mesh)

        # Create a lightweight evaluator-like object
        class _EntropyEvaluator:
            pass
        self.evaluator = _EntropyEvaluator()
        self.evaluator.mesh = mesh
        self.evaluator.boundary_conditions = bc

        # Cache flattened (1D) mesh arrays for the hot path.
        # Mesh stores (N,1) column vectors; we flatten once here.
        P_basic = np.asarray(mesh.basic_pressure).ravel()
        P_stag = np.asarray(mesh.staggered_pressure).ravel()
        self._area_flat = np.asarray(mesh.basic.area).ravel()
        self._volume_flat = np.asarray(mesh.basic.volume).ravel()
        self._r_basic_flat = np.asarray(mesh.basic.radii).ravel()
        self._P_stag_flat = P_stag
        self._P_basic_flat = P_basic
        # Gravity: try mesh EOS attribute first, then mesh settings
        g = abs(float(getattr(
            mesh.eos, '_gravitational_acceleration',
            self.parameters.mesh.gravitational_acceleration,
        )))

        # Create entropy phase evaluators for staggered and basic nodes
        phase_kwargs = dict(
            entropy_eos=self.entropy_eos,
            rheological_transition_melt_fraction=(
                self.parameters.phase_mixed.rheological_transition_melt_fraction
            ),
            rheological_transition_width=(
                self.parameters.phase_mixed.rheological_transition_width
            ),
            grain_size=self.parameters.phase_mixed.grain_size,
        )

        # Get viscosity from config
        try:
            phase_kwargs['viscosity_solid'] = float(self.parameters.phase_solid.viscosity.eval())
        except Exception:
            phase_kwargs['viscosity_solid'] = 1e21
        try:
            phase_kwargs['viscosity_liquid'] = float(self.parameters.phase_liquid.viscosity.eval())
        except Exception:
            phase_kwargs['viscosity_liquid'] = 1e-1

        # Wire thermal conductivity from config
        try:
            phase_kwargs['thermal_conductivity_solid'] = float(
                self.parameters.phase_solid.thermal_conductivity.eval())
        except Exception:
            pass  # use default 4.0 W/m/K
        try:
            phase_kwargs['thermal_conductivity_liquid'] = float(
                self.parameters.phase_liquid.thermal_conductivity.eval())
        except Exception:
            pass  # use default 2.0 W/m/K

        phase_stag = EntropyPhaseEvaluator(
            gravitational_acceleration=g,
            **phase_kwargs,
        )
        phase_stag.set_pressure(P_stag)

        phase_basic = EntropyPhaseEvaluator(
            gravitational_acceleration=g,
            **phase_kwargs,
        )
        phase_basic.set_pressure(P_basic)

        # Energy settings
        energy = self.parameters.energy
        self.state = EntropyState(
            evaluator=self.evaluator,
            phase_staggered=phase_stag,
            phase_basic=phase_basic,
            conduction=energy.conduction,
            convection=energy.convection,
            gravitational_separation=energy.gravitational_separation,
            mixing=energy.mixing,
            radionuclides=energy.radionuclides,
            tidal=energy.tidal,
            tidal_array=getattr(energy, 'tidal_array', [0.0]),
            eddy_diffusivity_thermal=getattr(energy, 'eddy_diffusivity_thermal', 1.0),
            eddy_diffusivity_chemical=energy.eddy_diffusivity_chemical,
            kappah_floor=energy.kappah_floor,
            bottom_up_grav_sep=getattr(energy, 'bottom_up_grav_sep', True),
        )
        # Store radionuclide data for the state's heating computation
        if hasattr(self.parameters, 'radionuclides'):
            self.evaluator.radionuclides = self.parameters.radionuclides
        else:
            self.evaluator.radionuclides = []

    def reset(self) -> None:
        """Reset for a new integration (PROTEUS coupling loop).

        Re-reads the EOS mesh file if eos_method=2, then rebuilds
        the mesh, BCs, and entropy state. Matches Solver.reset().
        """
        logger.info('Resetting EntropySolver')
        if self.parameters.mesh.eos_method == 2 and self.parameters.mesh.eos_file:
            arr = np.loadtxt(self.parameters.mesh.eos_file)
            self.parameters.mesh.eos_radius = arr[:, 0]
            self.parameters.mesh.eos_pressure = arr[:, 1]
            self.parameters.mesh.eos_density = arr[:, 2]
            self.parameters.mesh.eos_gravity = arr[:, 3]
        self._initialize_internals()

    def set_initial_entropy(self, S_init: npt.NDArray | float) -> None:
        """Set the initial entropy profile.

        Parameters
        ----------
        S_init : array or float
            Entropy at staggered nodes [J/kg/K]. If scalar, sets uniform
            (isentropic) profile.
        """
        n_stag = self.evaluator.mesh.staggered.radii.shape[0]
        if np.isscalar(S_init):
            self._S0 = np.full(n_stag, float(S_init))
        else:
            self._S0 = np.asarray(S_init, dtype=float)
            if len(self._S0) != n_stag:
                raise ValueError(
                    f'S_init length {len(self._S0)} != mesh staggered nodes {n_stag}'
                )
        logger.info('Initial entropy: S_min=%.0f, S_max=%.0f J/kg/K',
                     self._S0.min(), self._S0.max())

    def dSdt(
        self,
        time: npt.NDArray | float,
        entropy: npt.NDArray,
    ) -> npt.NDArray:
        """dS/dt at the staggered nodes.

        Supports vectorized evaluation: when entropy is (N, K), returns
        (N, K) by looping over K columns. This enables scipy BDF to
        evaluate multiple perturbations for Jacobian approximation.

        Parameters
        ----------
        time : float
            Time [yr].
        entropy : array
            Entropy at staggered nodes [J/kg/K]. Shape (N,) or (N, K).

        Returns
        -------
        array
            dS/dt at staggered nodes [J/kg/K/yr]. Same shape as input.
        """
        # Handle vectorized (N, K) input by looping over columns
        if entropy.ndim > 1:
            result = np.zeros_like(entropy)
            for k in range(entropy.shape[1]):
                result[:, k] = self._dSdt_single(time, entropy[:, k])
            return result
        return self._dSdt_single(time, entropy)

    def _dSdt_single(
        self,
        time: npt.NDArray | float,
        entropy: npt.NDArray,
    ) -> npt.NDArray:
        """dS/dt for a single entropy profile (1D)."""
        self.state.update(entropy, time)

        # Apply flux BCs directly (not via BC module, which expects 2D arrays).
        # Surface: grey-body or prescribed flux
        bc = self.evaluator.boundary_conditions._settings
        if bc.outer_boundary_condition == 1:
            # Grey-body: F = emissivity * sigma * (T_surf^4 - T_eq^4)
            T_surf = self.state.top_temperature.item()
            T_eq = bc.equilibrium_temperature
            self.state._heat_flux[-1] = (
                bc.emissivity * Stefan_Boltzmann * (T_surf**4 - T_eq**4)
            )
        elif bc.outer_boundary_condition == 4:
            # Prescribed flux (PROTEUS coupling)
            self.state._heat_flux[-1] = bc.outer_boundary_value

        # CMB boundary condition
        if bc.inner_boundary_condition == 1:
            # Core cooling (Bower+2018 Eq. 37, matching boundary.py):
            # alpha = (R_1/R_0)^2 / (1 + C_cell / (C_core * tfac))
            # Both capacities must be THERMAL (rho*Cp*V, units J/K),
            # not entropy (rho*T*V). Use Cp from the phase evaluator.
            r_cmb = float(self._r_basic_flat[0])
            core_cap = (
                4.0 / 3.0 * np.pi * r_cmb**3
                * self.evaluator.mesh.settings.core_density
                * bc.core_heat_capacity
            )
            rho_first = float(np.asarray(self.state.phase_staggered.density()).flat[0])
            cp_first = float(np.asarray(self.state.phase_staggered.heat_capacity()).flat[0])
            vol_first = float(self._volume_flat[0])
            cell_cap = vol_first * rho_first * cp_first  # J/K (thermal)
            r_above = float(self._r_basic_flat[1])
            tfac = getattr(bc, 'tfac_core_avg', 1.147)
            radius_ratio = r_above / r_cmb
            alpha = radius_ratio**2 / (cell_cap / (core_cap * tfac) + 1.0)
            self.state._heat_flux[0] = alpha * self.state._heat_flux[1]
        elif bc.inner_boundary_condition == 2:
            self.state._heat_flux[0] = bc.inner_boundary_value
        elif bc.inner_boundary_condition == 3:
            pass  # prescribed T
        else:
            self.state._heat_flux[0] = 0.0  # insulating

        # Flux divergence: dE/dr at staggered nodes.
        # Use cached 1D mesh arrays (area, volume) from _initialize_internals.
        energy_flux = self.state.heat_flux * self._area_flat
        delta_energy_flux = np.diff(energy_flux)

        # Capacitance: rho * T * volume (for entropy equation)
        cap = np.asarray(self.state.capacitance_staggered()).ravel()
        capacitance = cap * self._volume_flat

        # dS/dt from flux divergence [J/kg/K/s]
        dSdt = -delta_energy_flux / capacitance

        # Internal heating: dS/dt += H / T (SPIDER rhs.c line 62)
        # H is power per unit mass [W/kg], T is temperature [K]
        H = self.state.heating
        T_stag = np.asarray(self.state.phase_staggered.temperature()).ravel()
        dSdt += H / np.maximum(T_stag, 1.0)

        # Convert to J/kg/K/yr
        dSdt *= SECS_PER_YEAR

        return dSdt

    @property
    def solution(self) -> OptimizeResult:
        return self._solution

    @property
    def entropy_staggered(self) -> npt.NDArray:
        """Entropy at staggered nodes from the solution."""
        return self._solution.y

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature at staggered nodes (derived from S via EOS)."""
        S = self._solution.y[:, -1] if self._solution.y.ndim > 1 else self._solution.y
        P = self.evaluator.mesh.staggered.pressure
        return self.entropy_eos.temperature(P, S)

    def _build_jac_sparsity(self) -> 'scipy.sparse.spmatrix':
        """Build the Jacobian sparsity pattern for the BDF solver.

        The entropy equation couples node i to its nearest neighbours via
        the flux divergence operator (diff of fluxes at basic nodes i and
        i+1, each depending on staggered nodes i-1, i, i+1 through the
        mesh transform matrices). The bulk Jacobian is tridiagonal.

        At boundaries the d/dr transform uses a 3-point extrapolation
        stencil, extending the bandwidth by 1. A pentadiagonal pattern
        (bandwidth 2) safely covers all entries including boundaries.

        With this sparsity hint scipy groups finite-difference
        perturbations by graph colouring: 5 RHS evaluations instead of
        N+1 for the full dense Jacobian. For N=100 this is a ~20x
        reduction in Jacobian construction cost.
        """
        N = len(self._S0)
        offsets = [-2, -1, 0, 1, 2]
        data = [np.ones(N - abs(k)) for k in offsets]
        return sparse_diags(data, offsets, shape=(N, N), format='csc')

    def solve(self) -> None:
        """Run the BDF time integration."""
        start_time = self.parameters.solver.start_time
        end_time = self.parameters.solver.end_time
        atol = max(self.parameters.solver.atol, 0.01)  # entropy in J/kg/K
        rtol = self.parameters.solver.rtol

        logger.info(
            'EntropySolver: integrating from %.2e to %.2e yr (atol=%.2e, rtol=%.2e)',
            start_time, end_time, atol, rtol,
        )

        jac_sparsity = self._build_jac_sparsity()

        self._solution = solve_ivp(
            self.dSdt,
            (start_time, end_time),
            self._S0,
            method='BDF',
            vectorized=True,
            dense_output=True,
            atol=atol,
            rtol=rtol,
            jac_sparsity=jac_sparsity,
        )

        if self._solution.status == 0:
            logger.info('EntropySolver: integration completed successfully.')
            self.stop_early = False
        else:
            logger.error(
                'EntropySolver: integration failed (status=%d): %s',
                self._solution.status, self._solution.message,
            )
            self.stop_early = True

    def get_state(self) -> SolverOutput:
        """Extract the solver state as a clean output dataclass.

        This is the primary API for callers to retrieve results. It
        avoids the need to access solver internals (evaluator, mesh,
        state, phase objects).

        Returns
        -------
        SolverOutput
            Dataclass containing all quantities needed by PROTEUS.
        """
        sol = self._solution
        eos = self.entropy_eos
        mesh = self.evaluator.mesh

        S_final = sol.y[:, -1]
        P_stag = self._P_stag_flat
        r_basic = self._r_basic_flat
        r_stag = np.asarray(mesh.staggered.radii).ravel()
        vol = self._volume_flat

        T_stag = np.asarray(eos.temperature(P_stag, S_final)).ravel()
        phi_stag = np.asarray(eos.melt_fraction(P_stag, S_final)).ravel()
        rho_stag = np.asarray(eos.density(P_stag, S_final)).ravel()

        # Refresh the state at the final entropy for derived quantities
        self.state.update(S_final, sol.t[-1])
        visc_stag = np.asarray(self.state.phase_staggered.viscosity()).ravel()
        heat_flux = self.state.heat_flux.copy()
        heating = self.state.heating.copy()
        eddy_diff = self.state.eddy_diffusivity.copy()
        cap_stag = np.asarray(self.state.capacitance_staggered()).ravel()

        # Scalar quantities
        mass_stag = rho_stag * vol
        M_mantle = float(np.sum(mass_stag))
        T_magma = float(T_stag[-1])
        T_core = float(T_stag[0])
        Phi_global = float(np.dot(phi_stag, vol) / np.sum(vol))

        # Rheological front depth
        phi_rheo = self.parameters.phase_mixed.rheological_transition_melt_fraction
        phi_basic = mesh.quantity_at_basic_nodes(phi_stag).ravel()
        if Phi_global > 0.99:
            rf = float(r_basic[0])
        elif Phi_global < 0.01:
            rf = float(r_basic[-1])
        else:
            idx = np.argmin(np.abs(phi_basic - phi_rheo))
            rf = float(r_basic[idx])
        R_outer = float(r_basic[-1])
        RF_depth = 1.0 - rf / R_outer if R_outer > 0 else 0.0

        # Thermal energy (sensible, for comparison with SPIDER)
        CP_REF = 1200.0
        E_th = float(np.sum(mass_stag * CP_REF * T_stag))

        # Effective heat capacity
        Cp_eff = float(np.sum(cap_stag * vol)) / max(M_mantle, 1.0)

        # Volumetric melt fraction (porosity-based)
        rho_sol = np.asarray(
            eos._lookup_at_phase_boundary('density', P_stag, 'solid')
        ).ravel()
        rho_liq = np.asarray(
            eos._lookup_at_phase_boundary('density', P_stag, 'melt')
        ).ravel()
        rho_bulk = 1.0 / (
            phi_stag / np.where(rho_liq > 0, rho_liq, 1.0)
            + (1 - phi_stag) / np.where(rho_sol > 0, rho_sol, 1.0)
        )
        drho = rho_sol - rho_liq
        safe_drho = np.where(np.abs(drho) > 1e-10, drho, 1.0)
        porosity = np.clip((rho_sol - rho_bulk) / safe_drho, 0, 1)
        Phi_global_vol = float(np.sum(porosity * vol) / np.sum(vol))

        # Heating flux
        area_surf = 4 * np.pi * float(r_basic[-1]) ** 2
        F_heat_total = float(np.dot(heating, mass_stag)) / area_surf

        return SolverOutput(
            S_final=S_final,
            T_stag=T_stag,
            phi_stag=phi_stag,
            rho_stag=rho_stag,
            visc_stag=visc_stag,
            P_stag=P_stag,
            r_basic=r_basic,
            r_stag=r_stag,
            vol=vol,
            mass_stag=mass_stag,
            heat_flux=heat_flux,
            heating=heating,
            eddy_diff=eddy_diff,
            cap_stag=cap_stag,
            T_magma=T_magma,
            T_core=T_core,
            Phi_global=Phi_global,
            Phi_global_vol=Phi_global_vol,
            M_mantle=M_mantle,
            M_mantle_liquid=float(np.sum(phi_stag * mass_stag)),
            M_mantle_solid=float(M_mantle - np.sum(phi_stag * mass_stag)),
            RF_depth=RF_depth,
            E_th=E_th,
            Cp_eff=Cp_eff,
            F_heat_total=F_heat_total,
            dt_actual=float(sol.t[-1] - sol.t[0]),
            status=sol.status,
        )
