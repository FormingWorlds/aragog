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
        # The truth source for n_stag is mesh.staggered.radii — some
        # mesh builds give mesh.staggered_pressure a different shape
        # (e.g., one extra entry for boundary handling) so we slice
        # the pressure array to match.
        self._n_stag = int(np.asarray(mesh.staggered.radii).shape[0])
        P_basic = np.asarray(mesh.basic_pressure).ravel()
        P_stag = np.asarray(mesh.staggered_pressure).ravel()
        if P_stag.shape[0] != self._n_stag:
            logger.warning(
                'mesh.staggered_pressure length %d != mesh.staggered.radii '
                'length %d, slicing to the latter',
                P_stag.shape[0], self._n_stag,
            )
            P_stag = P_stag[:self._n_stag]
        self._area_flat = np.asarray(mesh.basic.area).ravel()
        self._volume_flat = np.asarray(mesh.basic.volume).ravel()
        self._r_basic_flat = np.asarray(mesh.basic.radii).ravel()
        self._P_stag_flat = P_stag
        self._P_basic_flat = P_basic

        # CMB BC mode (set here from config so dSdt can dispatch even
        # before set_initial_entropy is called):
        #   'quasi_steady' = production default (v3 alpha-factor BC),
        #     gives -19 % T_core gap to SPIDER on R8 CHILI but is
        #     stable and well-tested. See parent memory file for the
        #     known limitation.
        #   'bower2018' = EXPERIMENTAL: T_core as ODE state variable
        #     with conduction-only F_cmb. The conductive flux is
        #     ~5 OOM smaller than the actual physical CMB heat flow
        #     (which is dominated by convective coupling, not pure
        #     conduction). Result: T_core stays too hot. Needs a
        #     thermal-boundary-layer parameterization to be useful.
        #     Kept in the codebase for follow-up work.
        self._core_bc = getattr(
            self.parameters.boundary_conditions, 'core_bc', 'quasi_steady'
        )
        # Gravity: try mesh EOS attribute first, then mesh settings
        g = abs(float(getattr(
            mesh.eos, '_gravitational_acceleration',
            self.parameters.mesh.gravitational_acceleration,
        )))

        # Create entropy phase evaluators for staggered and basic nodes.
        # cp_blend selects how Cp is computed in the mushy zone:
        #   'latent' = SPIDER-parity v4 convention (latent-heat-augmented)
        #   'linear' = legacy v3 convention (linear blend of pure-phase Cp)
        cp_blend = getattr(self.parameters.phase_mixed, 'cp_blend', 'latent')

        phase_kwargs = dict(
            entropy_eos=self.entropy_eos,
            rheological_transition_melt_fraction=(
                self.parameters.phase_mixed.rheological_transition_melt_fraction
            ),
            rheological_transition_width=(
                self.parameters.phase_mixed.rheological_transition_width
            ),
            grain_size=self.parameters.phase_mixed.grain_size,
            cp_blend=cp_blend,
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
        """Set the initial entropy profile and (if used) initial T_core.

        Parameters
        ----------
        S_init : array or float
            Entropy at staggered nodes [J/kg/K]. If scalar, sets uniform
            (isentropic) profile.

        Notes
        -----
        When the v4 Bower core BC is active (``core_bc='bower2018'``,
        the default), the solver state vector is N+1 in length, with
        ``state[-1] == T_core``. The initial T_core is taken from the
        bottom-cell mantle temperature derived from S_init via the EOS,
        unless ``set_initial_core_temperature`` has been called first.
        For the legacy quasi-steady BC (``core_bc='quasi_steady'``)
        the state vector is N as before.
        """
        # Prefer the cached _n_stag from _initialize_internals; fall
        # back to the mesh accessor for legacy callers that bypass it.
        if hasattr(self, '_n_stag') and self._n_stag is not None:
            n_stag = self._n_stag
        else:
            n_stag = self.evaluator.mesh.staggered.radii.shape[0]
            self._n_stag = n_stag

        if np.isscalar(S_init):
            S_arr = np.full(n_stag, float(S_init))
        else:
            S_arr = np.asarray(S_init, dtype=float)
            if len(S_arr) != n_stag:
                raise ValueError(
                    f'S_init length {len(S_arr)} != mesh staggered nodes {n_stag}'
                )

        # Prefer the cached _core_bc from _initialize_internals.
        if hasattr(self, '_core_bc') and self._core_bc is not None:
            core_bc = self._core_bc
        else:
            core_bc = getattr(self.parameters.boundary_conditions,
                              'core_bc', 'quasi_steady')
            self._core_bc = core_bc

        if core_bc == 'bower2018':
            # State = [S_0, ..., S_{N-1}, T_core]
            # T_core_init priority order:
            #   1. user override via set_initial_core_temperature
            #   2. preserve from a previous solve (the v4 hot path
            #      between PROTEUS coupling steps -- without this the
            #      core enthalpy ODE gets reset on every reset() and
            #      the integrated T_core is lost)
            #   3. EOS-derived bottom-cell mantle T (cold-start init)
            T_core_init = getattr(self, '_T_core_init', None)
            if T_core_init is None:
                # Preserve from previous solution if available
                prev_sol = getattr(self, '_solution', None)
                if (prev_sol is not None
                        and getattr(prev_sol, 'y', None) is not None
                        and prev_sol.y.size > 0
                        and prev_sol.y.shape[0] == n_stag + 1):
                    T_core_init = float(prev_sol.y[n_stag, -1])
                    logger.info(
                        'Preserved T_core from previous solve: %.0f K',
                        T_core_init,
                    )
            if T_core_init is None:
                # Cold start: derive from bottom-cell mantle T via EOS
                P_bottom = float(self._P_stag_flat[0])
                T_core_init = float(np.asarray(
                    self.entropy_eos.temperature(
                        np.array([P_bottom]), np.array([S_arr[0]])
                    )
                ).item())
            self._S0 = np.empty(n_stag + 1)
            self._S0[:n_stag] = S_arr
            self._S0[n_stag] = T_core_init
            logger.info(
                'Initial state (v4 Bower core BC): S_min=%.0f, S_max=%.0f, '
                'T_core_init=%.0f K',
                S_arr.min(), S_arr.max(), T_core_init,
            )
        else:
            # Legacy v3 quasi-steady BC: state = [S_0, ..., S_{N-1}]
            self._S0 = S_arr
            logger.info('Initial entropy (v3 quasi-steady BC): '
                         'S_min=%.0f, S_max=%.0f J/kg/K',
                         S_arr.min(), S_arr.max())

    def set_initial_core_temperature(self, T_core_init: float) -> None:
        """Set the initial core temperature (v4 Bower BC only).

        Must be called BEFORE ``set_initial_entropy``. If not called,
        the initial T_core defaults to the bottom-cell mantle
        temperature derived from S_init via the EOS.
        """
        self._T_core_init = float(T_core_init)

    def dSdt(
        self,
        time: npt.NDArray | float,
        state_vec: npt.NDArray,
    ) -> npt.NDArray:
        """Time derivative of the full state vector.

        For the v4 Bower core BC the state vector is
        ``[S_0, ..., S_{N-1}, T_core]`` of length N+1, and this returns
        ``[dS/dt, dT_core/dt]`` of the same length.

        For the legacy quasi-steady BC the state vector is just
        ``[S_0, ..., S_{N-1}]`` of length N.

        Supports vectorized evaluation: when ``state_vec`` is 2D
        ``(N+1, K)`` (or ``(N, K)``), returns the same shape by looping
        over K columns. This enables scipy BDF to evaluate multiple
        perturbations for finite-difference Jacobian approximation.

        Parameters
        ----------
        time : float
            Time [yr].
        state_vec : array
            Solver state vector [J/kg/K for entropy, K for T_core].
            Shape (N,)/(N+1,) or (N,K)/(N+1,K).

        Returns
        -------
        array
            d(state_vec)/dt with the same shape as the input.
        """
        if state_vec.ndim > 1:
            result = np.zeros_like(state_vec)
            for k in range(state_vec.shape[1]):
                result[:, k] = self._dSdt_single(time, state_vec[:, k])
            return result
        return self._dSdt_single(time, state_vec)

    def _dSdt_single(
        self,
        time: npt.NDArray | float,
        state_vec: npt.NDArray,
    ) -> npt.NDArray:
        """Time derivative of one state vector column."""
        n_stag = self._n_stag
        bower = (self._core_bc == 'bower2018')

        if bower:
            entropy = state_vec[:n_stag]
            T_core = float(state_vec[n_stag])
        else:
            entropy = state_vec
            T_core = None

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
            if bower:
                # v4 Bower+2018 Eq. 37 core BC
                # F_cmb = -k_eff * (T_above - T_core) / dr_half
                # The factor -1 makes F_cmb positive when heat flows
                # from mantle to core (i.e., T_above > T_core), which
                # matches Aragog's heat flux sign convention (positive
                # outward, so negative when flowing into the core).
                #
                # Wait: Aragog convention is that positive flux at a
                # basic node means heat flows in the +r direction.
                # At the CMB (innermost basic node) this means positive
                # = outward (from core to mantle). So when T_core >
                # T_above, F[0] > 0 (heat flows core -> mantle, core
                # cooling). When T_core < T_above, F[0] < 0 (heat flows
                # mantle -> core, core heating).
                #
                # Discrete Fourier law:
                #   F[0] = -k_eff * dT/dr|cmb
                #        ≈ -k_eff * (T_above - T_core) / dr_half
                # where T_above is the bottom mantle cell T and dr_half
                # is the half-distance from cell 0 center to the CMB.
                # When T_above > T_core, dT/dr > 0, so F[0] < 0 (heat
                # flows inward = into core), which matches.
                T_above = float(np.asarray(
                    self.state.phase_staggered.temperature()
                ).flat[0])
                k_above = float(np.asarray(
                    self.state.phase_staggered.thermal_conductivity()
                ).flat[0]) if hasattr(self.state.phase_staggered,
                                      'thermal_conductivity') else 4.0
                r_cmb = float(self._r_basic_flat[0])
                r_above = float(self._r_basic_flat[1])
                dr_half = 0.5 * (r_above - r_cmb)
                F_cmb = -k_above * (T_above - T_core) / max(dr_half, 1.0)
                self.state._heat_flux[0] = F_cmb
            else:
                # Legacy v3 quasi-steady BC (alpha factor partition)
                r_cmb = float(self._r_basic_flat[0])
                core_cap = (
                    4.0 / 3.0 * np.pi * r_cmb**3
                    * self.evaluator.mesh.settings.core_density
                    * bc.core_heat_capacity
                )
                rho_first = float(np.asarray(
                    self.state.phase_staggered.density()).flat[0])
                cp_first = float(np.asarray(
                    self.state.phase_staggered.heat_capacity()).flat[0])
                vol_first = float(self._volume_flat[0])
                cell_cap = vol_first * rho_first * cp_first  # J/K
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
        energy_flux = self.state.heat_flux * self._area_flat
        delta_energy_flux = np.diff(energy_flux)

        # Capacitance: rho * T * volume (for entropy equation)
        cap = np.asarray(self.state.capacitance_staggered()).ravel()
        capacitance = cap * self._volume_flat

        # dS/dt from flux divergence [J/kg/K/s]
        dSdt = -delta_energy_flux / capacitance

        # Internal heating: dS/dt += H / T
        H = self.state.heating
        T_stag = np.asarray(self.state.phase_staggered.temperature()).ravel()
        dSdt += H / np.maximum(T_stag, 1.0)

        # Convert to J/kg/K/yr
        dSdt *= SECS_PER_YEAR

        if not bower:
            return dSdt

        # v4: append dT_core/dt = -F_cmb * area_cmb / (M_core * Cp_core)
        # F_cmb is positive outward (core -> mantle = core cooling).
        # When F_cmb > 0, the core LOSES heat at rate F_cmb * area_cmb,
        # so dT_core/dt < 0.
        F_cmb = float(self.state._heat_flux[0])
        r_cmb = float(self._r_basic_flat[0])
        area_cmb = 4.0 * np.pi * r_cmb**2
        M_core = (
            4.0 / 3.0 * np.pi * r_cmb**3
            * self.evaluator.mesh.settings.core_density
        )
        Cp_core = float(bc.core_heat_capacity)
        dT_core_dt = -F_cmb * area_cmb / max(M_core * Cp_core, 1.0)
        # Convert from K/s to K/yr
        dT_core_dt *= SECS_PER_YEAR

        return np.concatenate([dSdt, [dT_core_dt]])

    @property
    def solution(self) -> OptimizeResult | None:
        """Last solve_ivp result, or ``None`` if ``solve()`` has not been
        called yet. Returning ``None`` (instead of raising
        ``AttributeError``) matters for the PROTEUS JAX dispatch path,
        where ``AragogJAXRunner`` handles the actual integration and
        the scipy ``EntropySolver`` lives only to hold Parameters / BC
        state — its ``solve()`` is never invoked, so ``_solution`` is
        never set. Callers already handle ``sol is None``.
        """
        return getattr(self, '_solution', None)

    @property
    def entropy_staggered(self) -> npt.NDArray:
        """Entropy at staggered nodes from the solution.

        For the v4 Bower core BC, the solver state vector is N+1 in
        length (S followed by T_core); we strip the trailing T_core
        row and return only the entropy block.
        """
        y = self._solution.y
        if self._core_bc == 'bower2018':
            return y[:self._n_stag]
        return y

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature at staggered nodes (derived from S via EOS)."""
        y = self._solution.y
        if self._core_bc == 'bower2018':
            S = y[:self._n_stag, -1] if y.ndim > 1 else y[:self._n_stag]
        else:
            S = y[:, -1] if y.ndim > 1 else y
        P = self.evaluator.mesh.staggered.pressure
        return self.entropy_eos.temperature(P, S)

    def _build_jac_sparsity(self) -> 'scipy.sparse.spmatrix':
        """Build the Jacobian sparsity pattern for the BDF solver.

        The entropy equation couples node i to its nearest neighbours via
        the flux divergence operator. The bulk Jacobian is tridiagonal,
        extended to pentadiagonal at the boundaries for the 3-point
        d/dr extrapolation stencil.

        For the v4 Bower core BC, the state vector grows by one
        (T_core at index N) and the sparsity gets two extra entries:

          - row N (T_core) couples to S[0] (via T_above in F_cmb)
            and to itself
          - row 0 (S[0]) couples to T_core (via F_cmb feedback)

        With this sparsity hint scipy groups finite-difference
        perturbations by graph colouring, giving ~5 RHS evaluations
        per Jacobian instead of N+1.
        """
        from scipy.sparse import lil_matrix

        n_stag = self._n_stag
        bower = (self._core_bc == 'bower2018')
        N = n_stag + (1 if bower else 0)

        J = lil_matrix((N, N), dtype=float)
        # Pentadiagonal block for the entropy part
        for i in range(n_stag):
            for k in range(-2, 3):
                j = i + k
                if 0 <= j < n_stag:
                    J[i, j] = 1.0

        if bower:
            # T_core (row N) couples to S[0] and itself
            J[n_stag, 0] = 1.0
            J[n_stag, n_stag] = 1.0
            # S[0] (row 0) couples to T_core via F_cmb feedback
            J[0, n_stag] = 1.0

        return J.tocsc()

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

        n_stag = self._n_stag
        bower = (self._core_bc == 'bower2018')

        # Slice the final state vector. For v4 Bower BC the last
        # element is T_core; for legacy quasi-steady BC the whole
        # vector is the entropy profile.
        if bower:
            S_final = sol.y[:n_stag, -1]
            T_core_solver = float(sol.y[n_stag, -1])
        else:
            S_final = sol.y[:, -1]
            T_core_solver = None

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
        # Core temperature: prefer the v4 Bower-evolved value when
        # available; fall back to the bottom-cell mantle T (the v3
        # convention) otherwise.
        if T_core_solver is not None:
            T_core = T_core_solver
        else:
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

        # Thermal energy (sensible, for comparison with SPIDER).
        #
        # Use the real heat capacity Cp(P, S) from the EntropyEOS
        # phase evaluator. Until 2026-04-09 this used a hardcoded
        # CP_REF = 1200 J/kg/K, which under-counted E_th by ~25-30 %
        # at mantle conditions and produced a spurious +17 % offset
        # against SPIDER (which itself was also wrong, see the
        # parallel fix in proteus.interior_energetics.spider). With
        # both wrappers using their respective EOS Cp(P, S) values
        # the helpfile E_th is now physically meaningful and the
        # SPIDER/Aragog parity reduces to a true comparison.
        Cp_stag = np.asarray(self.state.phase_staggered.heat_capacity()).ravel()
        E_th = float(np.sum(mass_stag * Cp_stag * T_stag))

        # Effective heat capacity (mass-weighted mean Cp)
        Cp_eff = float(np.sum(mass_stag * Cp_stag)) / max(M_mantle, 1.0)

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
