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

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters
from aragog.solver.boundary import BoundaryConditions
from aragog.solver.entropy_state import EntropyState

# SUNDIALS CVODE via scikits.odes (the same solver SPIDER uses). scipy
# BDF/Radau collapse to machine-epsilon step size at the crystallisation
# front; CVODE handles the phase-transition stiffness cleanly.
try:
    from scikits_odes.ode import ode as _scikits_ode
    _CVODE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CVODE_AVAILABLE = False
    _scikits_ode = None  # type: ignore[assignment]

# Import SECS_PER_YEAR directly to avoid circular import with solver/__init__.py
from scipy import constants as _sp_constants
SECS_PER_YEAR: float = _sp_constants.Julian_year

logger = logging.getLogger(__name__)


def _phase_prop_float(raw, default):
    """Convert a phase-property config value to float.

    Parameters
    ----------
    raw
        Value from _PhaseParameters: a plain float (from PROTEUS) or
        a string expression with an .eval() method (legacy .cfg parser).
    default
        Fallback if conversion fails entirely.  None means no fallback.

    Returns
    -------
    float or None
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw.eval())
    except Exception:
        pass
    return default


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

    # Diagnostic decomposition at basic nodes (T_core phi=0 instability
    # investigation, see memory/tcore_phi0_*.md). Populated from the
    # final EntropyState after integration; only consumers that set
    # ``write_flux_diagnostics = true`` in the PROTEUS config read these.
    jcond_b: npt.NDArray        # conductive flux [W/m^2]
    jconv_b: npt.NDArray        # convective flux [W/m^2]
    jgrav_b: npt.NDArray        # grav-sep contribution to heat flux [W/m^2]
    jmix_b: npt.NDArray         # SPIDER-parity mixing flux [W/m^2]
    dSdr_b: npt.NDArray         # entropy gradient [J/kg/K/m]
    phi_basic: npt.NDArray      # EOS melt fraction at basic nodes [-]
    T_basic: npt.NDArray        # temperature at basic nodes [K]
    cp_basic: npt.NDArray       # heat capacity at basic nodes [J/kg/K]
    rho_basic: npt.NDArray      # density at basic nodes [kg/m^3]

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

    def __init__(self, parameters: Parameters, entropy_eos: EntropyEOS | None = None):
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
        self._r_stag_flat = np.asarray(mesh.staggered.radii).ravel()
        self._P_stag_flat = P_stag
        self._P_basic_flat = P_basic

        # CMB BC mode:
        #   'energy_balance' (default): dSdr_cmb is an ODE state variable
        #     evolved by the SPIDER-parity core energy balance (bc.c:76-131).
        #   'quasi_steady': alpha-factor heat-flux partition; FD-derived
        #     dSdr[0] at the boundary. Less robust at the crystallisation
        #     front and not bit-parity with SPIDER.
        self._core_bc = getattr(
            self.parameters.boundary_conditions, 'core_bc', 'energy_balance'
        )
        # Gravity: try mesh EOS attribute first, then mesh settings
        g = abs(float(getattr(
            mesh.eos, '_gravitational_acceleration',
            self.parameters.mesh.gravitational_acceleration,
        )))

        # Create entropy phase evaluators for staggered and basic nodes.
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

        # Get viscosity and thermal conductivity from config.
        # Values can be plain floats (from PROTEUS) or string expressions
        # (from Aragog's legacy .cfg parser with .eval()).  Try float()
        # first; fall back to .eval() for legacy strings.
        phase_kwargs['viscosity_solid'] = _phase_prop_float(
            self.parameters.phase_solid.viscosity, 1e21)
        phase_kwargs['viscosity_liquid'] = _phase_prop_float(
            self.parameters.phase_liquid.viscosity, 1e-1)
        # matprop_smooth_width: SPIDER's phase-boundary smoothing for
        # material properties. Read from the mixed-phase config if
        # available; defaults to 0 (no smoothing) for backward compat.
        phase_kwargs['matprop_smooth_width'] = float(getattr(
            self.parameters.phase_mixed, 'matprop_smooth_width', 0.0))
        cond_s = _phase_prop_float(
            self.parameters.phase_solid.thermal_conductivity, None)
        if cond_s is not None:
            phase_kwargs['thermal_conductivity_solid'] = cond_s
        cond_l = _phase_prop_float(
            self.parameters.phase_liquid.thermal_conductivity, None)
        if cond_l is not None:
            phase_kwargs['thermal_conductivity_liquid'] = cond_l

        # Constant-properties mode (SPIDER -use_const_properties parity)
        _const = getattr(self.parameters.phase_mixed, 'const_properties', False)
        if _const:
            phase_kwargs['const_properties'] = True
            for k in ('const_rho', 'const_Cp', 'const_alpha', 'const_cond',
                       'const_log10visc', 'const_T_ref', 'const_S_ref'):
                phase_kwargs[k] = float(getattr(self.parameters.phase_mixed, k))

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
            dilatation=getattr(energy, 'dilatation', False),
            tidal=energy.tidal,
            tidal_array=getattr(energy, 'tidal_array', [0.0]),
            eddy_diffusivity_thermal=getattr(energy, 'eddy_diffusivity_thermal', 1.0),
            eddy_diffusivity_chemical=energy.eddy_diffusivity_chemical,
            kappah_floor=energy.kappah_floor,
            bottom_up_grav_sep=getattr(energy, 'bottom_up_grav_sep', True),
            phase_smoothing=getattr(energy, 'phase_smoothing', 'cubic_hermite'),
        )
        # Store radionuclide data for the state's heating computation
        if hasattr(self.parameters, 'radionuclides'):
            self.evaluator.radionuclides = self.parameters.radionuclides
        else:
            self.evaluator.radionuclides = []

        # Tier 1 speedup (Step 1A): cache constant BC and mesh terms
        # so the dSdt hot path doesn't recompute them on every RHS
        # call. None of these depend on entropy.
        self._cache_bc_constants()

        # Nondimensionalisation scales matching SPIDER (spider.py:832-835).
        # After dividing by these, S_nd ~ O(1) and dSdr_nd ~ O(1),
        # so CVODE can use tight atol without excessive internal substeps.
        # In physical units S ~ 3000 J/kg/K, meaning atol=1e-10 demands
        # ~13 significant digits on a 4-digit number. With nondim,
        # atol=1e-10 needs only ~10 digits on an O(1) number.
        self._S_ref = 2993.025100070677  # entropy0 [J/kg/K] (spider.py:835)
        self._t_ref_yr = 1e5 / SECS_PER_YEAR  # time0 [yr]
        self._r_ref = 6.371e7        # radius0 [m]
        self._dSdr_ref = self._S_ref / self._r_ref  # [J/kg/K/m]

    def _cache_bc_constants(self) -> None:
        """Pre-compute BC and mesh constants for the dSdt hot path.

        Lifts core_density, core_heat_capacity, area_cmb, M_core,
        r_cmb, r_above, dr_half, vol_first, tfac_core_avg, and the
        outer/inner BC dispatch keys out of the per-RHS-call path.
        Called from ``_initialize_internals`` after BC + mesh setup.

        Tier 1 speedup, 2026-04-09. The dSdt loop previously
        recomputed all of these on every RHS evaluation, which
        meant ~10 µs/call of pure dispatch + arithmetic overhead.
        At ~1000 RHS calls per coupling step on R8 CHILI, that's
        ~10 ms per coupling step pure overhead, fully removable.
        """
        bc = self.evaluator.boundary_conditions._settings
        mesh = self.evaluator.mesh

        # Mesh geometry at the CMB
        r_cmb = float(self._r_basic_flat[0])
        r_above = float(self._r_basic_flat[1])
        self._cmb_r_cmb = r_cmb
        self._cmb_r_above = r_above
        self._cmb_dr_cmb = r_above - r_cmb          # basic-node spacing at CMB
        self._cmb_dr_half = 0.5 * self._cmb_dr_cmb  # basic-to-staggered half-spacing
        self._cmb_area = 4.0 * np.pi * r_cmb**2
        self._cmb_vol_first = float(self._volume_flat[0])

        # Core properties (constant in time)
        self._core_density = float(mesh.settings.core_density)
        self._core_cp = float(bc.core_heat_capacity)
        self._core_M = (4.0 / 3.0) * np.pi * r_cmb**3 * self._core_density
        self._core_cap = self._core_M * self._core_cp  # J/K
        self._core_tfac = float(getattr(bc, 'tfac_core_avg', 1.147))

        # Quasi-steady BC alpha factor uses (R_above/R_cmb)^2
        self._cmb_radius_ratio_sq = (r_above / r_cmb) ** 2

        # BC dispatch keys captured once
        self._outer_bc_kind = int(bc.outer_boundary_condition)
        self._outer_bc_value = float(bc.outer_boundary_value)
        self._outer_bc_emiss = float(bc.emissivity)
        self._outer_bc_T_eq = float(bc.equilibrium_temperature)
        self._inner_bc_kind = int(bc.inner_boundary_condition)
        self._inner_bc_value = float(bc.inner_boundary_value)

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

        Notes
        -----
        For ``core_bc='energy_balance'`` the state vector is length N+1,
        with ``state[-1] = dSdr_cmb`` (CMB entropy gradient evolved by
        the SPIDER bc.c:76-131 energy balance). For ``core_bc='gradient'``
        the state is length N+2 with ``[dS/dr at N+1 basic nodes, S_surf]``.
        For ``core_bc='quasi_steady'`` the state is length N (entropy only).
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
        logger.debug('set_initial_entropy: core_bc=%r, n_stag=%d', core_bc, n_stag)

        if core_bc == 'energy_balance':
            # SPIDER-parity core BC. State = [S_0..S_{N-1}, dSdr_cmb].
            # dSdr_cmb is the CMB-basic entropy gradient (mirror of
            # SPIDER's dSdxi[ind_cmb]); its time derivative is set by
            # the bc.c:76-131 formula in _dSdt_single from the actual
            # physical heat flux. Hot-start preserves the boundary
            # state across PROTEUS coupling resets.
            dSdr_cmb_init = getattr(self, '_dSdr_cmb_init', None)
            if dSdr_cmb_init is None:
                # Hot start: preserve from previous solution if shape matches
                prev_sol = getattr(self, '_solution', None)
                if (prev_sol is not None
                        and getattr(prev_sol, 'y', None) is not None
                        and prev_sol.y.size > 0
                        and prev_sol.y.shape[0] == n_stag + 1):
                    dSdr_cmb_init = float(prev_sol.y[n_stag, -1])
                    logger.info(
                        'Preserved dSdr_cmb from previous solve: %.3e J/kg/K/m',
                        dSdr_cmb_init,
                    )
            if dSdr_cmb_init is None:
                # Cold start: one-sided FD of S_init at the bottom.
                # dSdr_cmb ≈ (S_stag[1] - S_stag[0]) / (r_stag[1] - r_stag[0])
                # For a uniform S_init this is exactly zero, which
                # is the correct neutral-buoyancy starting point.
                if n_stag >= 2:
                    r_basic = np.asarray(
                        self.evaluator.mesh.basic.radii
                    ).ravel()
                    r_stag_0 = 0.5 * (r_basic[0] + r_basic[1])
                    r_stag_1 = 0.5 * (r_basic[1] + r_basic[2])
                    dSdr_cmb_init = (
                        (float(S_arr[1]) - float(S_arr[0]))
                        / max(r_stag_1 - r_stag_0, 1.0)
                    )
                else:
                    dSdr_cmb_init = 0.0
                logger.info(
                    'Cold-start dSdr_cmb from FD: %.3e J/kg/K/m',
                    dSdr_cmb_init,
                )
            self._S0 = np.empty(n_stag + 1)
            self._S0[:n_stag] = S_arr
            self._S0[n_stag] = float(dSdr_cmb_init)
            logger.info(
                'Initial state (energy_balance): S_min=%.0f, S_max=%.0f, '
                'dSdr_cmb_init=%.3e',
                S_arr.min(), S_arr.max(), dSdr_cmb_init,
            )
        elif core_bc == 'gradient':
            # Gradient-based formulation mirroring SPIDER's dS/dxi state.
            # State = [dS/dr at N+1 basic nodes, S_surf], length N+2.
            # S at staggered nodes is reconstructed by cumulative sum
            # at each RHS evaluation, providing the global smoothing
            # that prevents BDF overshoot at the solidus.
            mesh = self.evaluator.mesh
            dSdr_init = mesh.d_dr_at_basic_nodes(S_arr).ravel()
            n_basic = len(dSdr_init)
            S_surf = float(S_arr[-1])
            self._S0 = np.empty(n_basic + 1)
            self._S0[:n_basic] = dSdr_init
            self._S0[n_basic] = S_surf
            # Verify roundtrip
            S_check, _ = self._reconstruct_entropy(dSdr_init, S_surf)
            roundtrip_err = float(np.max(np.abs(S_check - S_arr)))
            logger.info(
                'Initial state (gradient): dSdr range [%.3e, %.3e] J/kg/K/m, '
                'S_surf=%.1f, roundtrip err=%.2e J/kg/K',
                dSdr_init.min(), dSdr_init.max(), S_surf, roundtrip_err,
            )
        else:
            # quasi_steady: state = [S_0, ..., S_{N-1}]
            self._S0 = S_arr
            logger.info('Initial entropy (quasi_steady): '
                         'S_min=%.0f, S_max=%.0f J/kg/K',
                         S_arr.min(), S_arr.max())

    def set_initial_dSdr_cmb(self, dSdr_cmb_init: float) -> None:
        """Set the initial CMB entropy gradient (Path A energy_balance only).

        Must be called BEFORE ``set_initial_entropy``. If not called,
        the initial ``dSdr_cmb`` is taken from the previous solution
        (if any), else from a one-sided FD of the staggered S_init at
        the bottom (which is zero for a uniform isentrope).
        """
        self._dSdr_cmb_init = float(dSdr_cmb_init)

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

        Tier 1 Step 1B (2026-04-09): the previous version had a
        ``vectorized=True`` dispatch branch that fell through to a
        sequential Python loop over the K columns -- pure overhead
        with zero scipy benefit. Removed; the solver now uses
        ``vectorized=False`` and the 1D path only.

        Parameters
        ----------
        time : float
            Time [yr].
        state_vec : array
            Solver state vector [J/kg/K for entropy, K for T_core].
            Shape (N,) or (N+1,) only.

        Returns
        -------
        array
            d(state_vec)/dt with the same shape as the input.
        """
        return self._dSdt_single(time, state_vec)

    def _dSdt_single(
        self,
        time: npt.NDArray | float,
        state_vec: npt.NDArray,
    ) -> npt.NDArray:
        """Time derivative of one state vector column.

        Four CMB BC modes:

        - 'quasi_steady' (default): state = [S_0, ..., S_{N-1}],
          length N. F_cmb is set by the alpha-factor partition of
          F[1].
        - 'energy_balance' (Path A SPIDER bit-parity): state =
          [S_0, ..., S_{N-1}, dSdr_cmb], length N+1. The boundary
          state dSdr_cmb is passed into ``state.update`` so the
          convective+conductive flux at the CMB basic node uses
          the boundary entropy gradient. d/dt(dSdr_cmb) is computed
          from SPIDER's bc.c:76-131 formula.
        - 'gradient': state = [dS/dr at N+1 basic nodes, S_surf],
          length N+2. Mirrors SPIDER's dS/dxi formulation. S at
          staggered nodes is reconstructed by cumulative sum from
          the surface inward at each RHS evaluation.
        """
        n_stag = self._n_stag
        gradient_mode = (self._core_bc == 'gradient')
        energy_balance = (self._core_bc == 'energy_balance')
        is_extended = energy_balance

        # ── Gradient mode: reconstruct S from the gradient state ──
        if gradient_mode:
            n_basic = n_stag + 1
            dSdr_basic = state_vec[:n_basic]
            S_surf = float(state_vec[n_basic])
            entropy, S_basic = self._reconstruct_entropy(dSdr_basic, S_surf)
            self.state.update(entropy, time, dSdr=dSdr_basic, entropy_basic=S_basic)
        elif is_extended:
            entropy = state_vec[:n_stag]
            extra = float(state_vec[n_stag])
        else:
            entropy = state_vec
            extra = None

        # In energy_balance mode the boundary state IS the entropy
        # gradient, and we pass it through to state.update() so
        # the flux operator at the CMB basic node uses the
        # boundary value rather than the FD-derived estimate.
        if gradient_mode:
            pass  # state.update already called above with dSdr
        elif energy_balance:
            self.state.update(entropy, time, dSdr_cmb=extra)
        else:
            self.state.update(entropy, time)

        # Apply flux BCs directly (not via BC module, which expects 2D arrays).
        # All BC dispatch keys and constants are pre-cached in
        # _cache_bc_constants(); see Tier 1 Step 1A.

        # Surface: grey-body or prescribed flux
        if self._outer_bc_kind == 1:
            # Grey-body: F = emissivity * sigma * (T_surf^4 - T_eq^4)
            T_surf = self.state.top_temperature.item()
            self.state._heat_flux[-1] = (
                self._outer_bc_emiss * Stefan_Boltzmann
                * (T_surf**4 - self._outer_bc_T_eq**4)
            )
        elif self._outer_bc_kind == 4:
            # Prescribed flux (PROTEUS coupling). Note: outer_boundary_value
            # is updated in setup_or_update_solver between coupling steps,
            # so re-read from the live BC object instead of the cache.
            self.state._heat_flux[-1] = float(
                self.evaluator.boundary_conditions._settings.outer_boundary_value
            )

        # CMB boundary condition
        if self._inner_bc_kind == 1:
            if gradient_mode or energy_balance:
                # heat_flux[0] is the physical flux computed from the
                # state-provided dS/dr at the CMB; nothing to override.
                pass
            else:
                # quasi_steady: alpha-factor heat-flux partition.
                rho_first = float(np.asarray(
                    self.state.phase_staggered.density()).flat[0])
                cp_first = float(np.asarray(
                    self.state.phase_staggered.heat_capacity()).flat[0])
                cell_cap = self._cmb_vol_first * rho_first * cp_first  # J/K
                alpha = self._cmb_radius_ratio_sq / (
                    cell_cap / (self._core_cap * self._core_tfac) + 1.0
                )
                self.state._heat_flux[0] = alpha * self.state._heat_flux[1]
        elif self._inner_bc_kind == 2:
            self.state._heat_flux[0] = self._inner_bc_value
        elif self._inner_bc_kind == 3:
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

        # ── Gradient mode: build d/dt(dS/dr) at basic nodes ──
        if gradient_mode:
            r_stag = self._r_stag_flat
            dr_stag = np.diff(r_stag)  # shape (N-1,)

            # Interior basic nodes (i=1..N-1): SPIDER rhs.c:71-72
            # d/dt(dS/dr)[i] = (dSdt_stag[i] - dSdt_stag[i-1]) / dr_stag[i-1]
            rhs_interior = np.diff(dSdt) / dr_stag  # shape (N-1,)

            # CMB basic node (i=0): core energy balance (SPIDER bc.c:113-122)
            F_cmb = float(self.state._heat_flux[0])
            T_cmb = float(np.asarray(
                self.state.phase_basic.temperature()
            ).flat[0])
            cp_cmb = float(np.asarray(
                self.state.phase_basic.heat_capacity()
            ).flat[0])
            dSdt_s_cmb_per_s = float(dSdt[0]) / SECS_PER_YEAR
            rhs_cmb_per_s = self._energy_balance_rhs_per_s(
                F_cmb_basic=F_cmb,
                dSdt_s_cmb_per_s=dSdt_s_cmb_per_s,
                T_cmb_basic=T_cmb,
                cp_cmb_basic=cp_cmb,
            )
            rhs_cmb = rhs_cmb_per_s * SECS_PER_YEAR

            # Surface basic node (i=N): copy from adjacent interior
            # (SPIDER bc.c:63 convention)
            rhs_surf = rhs_interior[-1]

            # dS_surf/dt = dSdt at the surface staggered node
            dS_surf_dt = dSdt[-1]

            # Assemble: [rhs_cmb, rhs_interior(N-1), rhs_surf, dS_surf_dt]
            n_basic = n_stag + 1
            rhs = np.empty(n_basic + 1)
            rhs[0] = rhs_cmb
            rhs[1:n_basic - 1] = rhs_interior
            rhs[n_basic - 1] = rhs_surf
            rhs[n_basic] = dS_surf_dt
            return rhs

        if not is_extended:
            return dSdt

        if energy_balance:
            # Path A boundary-state ODE (extracted to a pure helper
            # `_energy_balance_rhs_per_s` for unit testing). The inputs are
            # the F_cmb heat flux computed by state.update() above
            # (which used the boundary dSdr_cmb to set the CMB basic
            # node entropy), and the dS/dt at the bottom staggered
            # cell (dSdt[0], which we just built from the flux
            # divergence).
            F_cmb_basic = float(self.state._heat_flux[0])
            T_cmb_basic = float(np.asarray(
                self.state.phase_basic.temperature()
            ).flat[0])
            cp_cmb_basic = float(np.asarray(
                self.state.phase_basic.heat_capacity()
            ).flat[0])
            dSdt_s_cmb_per_s = float(dSdt[0]) / SECS_PER_YEAR

            d_dSdr_cmb_dt_per_s = self._energy_balance_rhs_per_s(
                F_cmb_basic=F_cmb_basic,
                dSdt_s_cmb_per_s=dSdt_s_cmb_per_s,
                T_cmb_basic=T_cmb_basic,
                cp_cmb_basic=cp_cmb_basic,
            )
            d_dSdr_cmb_dt = d_dSdr_cmb_dt_per_s * SECS_PER_YEAR

            return np.concatenate([dSdt, [d_dSdr_cmb_dt]])

        return dSdt

    def _energy_balance_rhs_per_s(
        self,
        F_cmb_basic: float,
        dSdt_s_cmb_per_s: float,
        T_cmb_basic: float,
        cp_cmb_basic: float,
    ) -> float:
        """SPIDER bc.c:76-131 rhs for d(dSdr_cmb)/dt at the CMB.

        Pure numerical helper extracted from ``_dSdt_single`` so the
        bit-parity formula can be unit-tested without building a full
        solver state. Mirrors SPIDER's C code:

            fac_cmb = cp_cmb / (cp_core * T_cmb * tfac * M_core)
            rhs     = (-Etot_cmb + Ecore) * fac_cmb - dSdt_s_cmb
            rhs    *= 2 / dr_cmb

        where ``Etot_cmb = F_cmb_basic * area_cmb`` is the heat flow
        at the CMB basic node, ``Ecore`` is the core internal heat
        source (zero for the simple core-cooling mode used here), and
        ``dr_cmb = r_basic[1] - r_basic[0]`` is the half-cell spacing
        between the CMB basic node and the first staggered node.

        All cached constants (``_core_cp``, ``_core_tfac``, ``_core_M``,
        ``_cmb_area``, ``_cmb_dr_cmb``) are taken from
        ``_cache_bc_constants()``.

        Parameters
        ----------
        F_cmb_basic : float
            Heat flux at the CMB basic node [W/m^2]. Positive = heat
            flowing OUT of the core. This is normally the
            state.update()-derived flux, which uses the boundary
            entropy gradient to set the bottom mantle cell value.
        dSdt_s_cmb_per_s : float
            dS/dt at the bottom staggered cell (cell adjacent to the
            CMB) in J/(kg*K*s). This is ``dSdt[0]`` from the flux
            divergence, converted back from per-year to per-second.
        T_cmb_basic : float
            Temperature at the CMB basic node [K], derived from the
            boundary-entropy via the EOS.
        cp_cmb_basic : float
            Heat capacity at the CMB basic node [J/(kg*K)], derived
            from the boundary-entropy via the EOS.

        Returns
        -------
        float
            d(dSdr_cmb)/dt in J/(kg*K*m*s). The caller is responsible
            for the per-year conversion (multiply by SECS_PER_YEAR).

        Notes
        -----
        The formula is linear in F_cmb_basic and dSdt_s_cmb_per_s, so
        the zero-flux-plus-zero-dSdt limit trivially returns zero.
        This is used as a smoke-test invariant.
        """
        # Etot_cmb in J/s (= W)
        E_tot_cmb = F_cmb_basic * self._cmb_area

        # fac_cmb in 1/(kg*K). SPIDER bc.c:113.
        fac_cmb = cp_cmb_basic / (
            self._core_cp * T_cmb_basic * self._core_tfac * self._core_M
        )

        # E_core = 0 for simple core cooling (no internal heat source).
        E_core = 0.0

        # dS_basic_cmb/dt from energy balance [J/(kg*K*s)]
        dS_basic_cmb_dt = (-E_tot_cmb + E_core) * fac_cmb

        # d/dt(dS/dr) at the CMB basic node, from the one-sided
        # centered difference between basic and staggered:
        #
        #   d/dt(dS/dr) = (dSdt_stag - dSdt_basic) / (dr/2)
        #               = (dSdt_stag - dSdt_basic) * 2 / dr
        #
        # SPIDER bc.c writes the algebraically equivalent form
        #     rhs = (dSdt_basic - dSdt_stag) * 2 / (xi_cmb - xi_above)
        # where (xi_cmb - xi_above) < 0 because SPIDER's xi
        # coordinate decreases from surface to CMB. The negative
        # denominator flips the sign to match the formula above.
        # Aragog's radii increase from CMB to surface, so dr > 0
        # and the numerator must be (stag - basic) explicitly.
        return (
            (dSdt_s_cmb_per_s - dS_basic_cmb_dt)
            * 2.0 / self._cmb_dr_cmb
        )

    def _reconstruct_entropy(
        self,
        dSdr_basic: npt.NDArray,
        S_surf: float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Reconstruct S at staggered and basic nodes from dS/dr gradients.

        Integrates from the surface inward, mirroring SPIDER's
        ``set_entropy_reconstruction_from_ctx`` (util.c:85-126).

        Parameters
        ----------
        dSdr_basic : array, shape (N+1,)
            dS/dr at basic nodes (index 0 = CMB, N = surface).
        S_surf : float
            Entropy at the outermost staggered node (index N-1).

        Returns
        -------
        S_stag : array, shape (N,)
            Reconstructed entropy at staggered nodes.
        S_basic : array, shape (N+1,)
            Reconstructed entropy at basic nodes.
        """
        r_basic = self._r_basic_flat
        r_stag = self._r_stag_flat
        n_stag = len(r_stag)
        n_basic = len(r_basic)

        # Staggered nodes: integrate from surface (index N-1) inward.
        # dSdr[i+1] at basic node i+1 (between stag[i] and stag[i+1])
        # gives S[i] = S[i+1] - dSdr[i+1] * (r_stag[i+1] - r_stag[i]).
        S_stag = np.empty(n_stag)
        S_stag[-1] = S_surf
        dr_stag = np.diff(r_stag)  # shape (N-1,)
        for i in range(n_stag - 2, -1, -1):
            S_stag[i] = S_stag[i + 1] - dSdr_basic[i + 1] * dr_stag[i]

        # Basic nodes: half-cell interpolation from adjacent staggered.
        S_basic = np.empty(n_basic)
        for i in range(1, n_basic - 1):
            S_basic[i] = S_stag[i - 1] + dSdr_basic[i] * (r_basic[i] - r_stag[i - 1])
        # Boundary extrapolation
        S_basic[0] = S_stag[0] + dSdr_basic[0] * (r_basic[0] - r_stag[0])
        S_basic[-1] = S_stag[-1] + dSdr_basic[-1] * (r_basic[-1] - r_stag[-1])

        return S_stag, S_basic

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
    def _state_is_extended(self) -> bool:
        """True when the state vector has more than N elements.

        - energy_balance: N+1 (entropy + dSdr_cmb)
        - gradient: N+2 (N+1 gradients + S_surf)
        - quasi_steady: N (entropy only)
        """
        return self._core_bc in ('energy_balance', 'gradient')

    @property
    def entropy_staggered(self) -> npt.NDArray:
        """Entropy at staggered nodes from the solution.

        For energy_balance the state vector is N+1; strip the trailing
        dSdr_cmb element. For gradient mode, reconstruct S from the
        gradient state.
        """
        y = self._solution.y
        if self._core_bc == 'gradient':
            n_basic = self._n_stag + 1
            dSdr = y[:n_basic]
            S_surf_arr = y[n_basic]
            S_surf = float(S_surf_arr[-1]) if S_surf_arr.ndim > 0 else float(S_surf_arr)
            S_stag, _ = self._reconstruct_entropy(dSdr[:, -1] if dSdr.ndim > 1 else dSdr, S_surf)
            return S_stag.reshape(-1, 1) if y.ndim > 1 else S_stag
        if self._state_is_extended:
            return y[:self._n_stag]
        return y

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature at staggered nodes (derived from S via EOS)."""
        y = self._solution.y
        if self._core_bc == 'gradient':
            S_stag = self.entropy_staggered
            S = S_stag[:, -1] if S_stag.ndim > 1 else S_stag
        elif self._state_is_extended:
            S = y[:self._n_stag, -1] if y.ndim > 1 else y[:self._n_stag]
        else:
            S = y[:, -1] if y.ndim > 1 else y
        if self.entropy_eos is not None:
            P = self.evaluator.mesh.staggered.pressure
            return self.entropy_eos.temperature(P, S)
        # const_properties: T = T_ref * exp((S - S_ref) / Cp)
        pm = self.parameters.phase_mixed
        return pm.const_T_ref * np.exp((S - pm.const_S_ref) / pm.const_Cp)

    def _build_jac_sparsity(self):
        """Build the Jacobian sparsity pattern for the BDF solver.

        The entropy equation couples node i to its nearest neighbours
        via the flux divergence operator. The bulk Jacobian is
        tridiagonal, extended to pentadiagonal at the boundaries
        for the 3-point d/dr extrapolation stencil.

        For energy_balance the state vector grows by one element
        (dSdr_cmb) and the Jacobian gets an extra row/column:

          - row N (dSdr_cmb) couples to S[0], S[1] (via the flux
            operator extension) and itself
          - rows 0 and 1 (S[0] and S[1]) gain couplings to dSdr_cmb
            via the boundary-flux feedback

        With this sparsity hint scipy groups finite-difference
        perturbations by graph colouring, giving ~5 RHS evaluations
        per Jacobian instead of N+1.
        """
        from scipy.sparse import lil_matrix

        # Gradient mode: the reconstruction couples all gradients to all
        # staggered nodes, making the Jacobian dense. Use no sparsity
        # hint (scipy will do full finite-difference Jacobian).
        if self._core_bc == 'gradient':
            return None

        n_stag = self._n_stag
        is_ext = self._state_is_extended
        N = n_stag + (1 if is_ext else 0)

        J = lil_matrix((N, N), dtype=float)
        # Pentadiagonal block for the entropy part
        for i in range(n_stag):
            for k in range(-2, 3):
                j = i + k
                if 0 <= j < n_stag:
                    J[i, j] = 1.0

        if is_ext:
            extra = n_stag  # index of the boundary state
            # Extra state couples to S[0] and S[1] (and S[2] for
            # energy_balance's pentadiagonal reach) and itself.
            J[extra, 0] = 1.0
            J[extra, 1] = 1.0
            if n_stag >= 3:
                J[extra, 2] = 1.0
            J[extra, extra] = 1.0
            # S[0], S[1] gain couplings to the extra state via
            # boundary-flux feedback.
            J[0, extra] = 1.0
            J[1, extra] = 1.0

        return J.tocsc()

    def _solve_cvode(
        self,
        start_time: float,
        end_time: float,
        y0: npt.NDArray,
        atol: float | npt.NDArray,
        rtol: float,
        max_step: float,
        rhs: 'Callable | None' = None,
    ) -> 'OptimizeResult':
        """Integrate the entropy equation using SUNDIALS CVODE.

        Returns a scipy.integrate.OdeResult-compatible OptimizeResult
        so the downstream extraction code (``sol.y``, ``sol.t``,
        ``sol.status``, ``sol.nfev``, ``sol.message``) is unchanged.

        CVODE uses the same BDF family as scipy, but the C
        implementation has a modified-Newton nonlinear solver with
        cached Jacobian factorisation and adaptive order 1..5, so it
        handles the crystallisation-front phase transition without
        collapsing its step size to machine epsilon. SPIDER uses the
        same CVODE (via PETSc's TSSUNDIALS interface).

        Parameters
        ----------
        rhs : callable or None
            If provided, used as the RHS function ``rhs(t, y) -> dydt``
            instead of ``self.dSdt``. Used by ``solve()`` to pass
            the nondimensionalised RHS wrapper.
        """
        # Zero-span edge case: scipy's solve_ivp accepts
        # `t_span = (t0, t0)` and returns the initial state trivially,
        # but CVODE rejects it with "tout too close to t0". This
        # happens during Aragog's static-init PROTEUS iterations where
        # start_time == end_time == 0. Return a scipy-compatible
        # result without invoking CVODE at all.
        if not (end_time > start_time):
            result = OptimizeResult()
            y_arr = np.asarray(y0, dtype=float).ravel().reshape(-1, 1)
            result.t = np.array([float(start_time)])
            result.y = y_arr
            result.nfev = 0
            result.status = 0
            result.message = 'zero-span solve, returned initial state'
            return result

        # Wrap the RHS into the in-place (t, y, ydot) -> int signature
        # that scikits.odes CVODE expects. Count RHS calls for parity
        # with scipy's sol.nfev.
        _rhs = rhs if rhs is not None else self.dSdt
        nfev_box = [0]

        def rhs_fn(t: float, y: npt.NDArray, ydot: npt.NDArray) -> int:
            ydot[:] = _rhs(t, y)
            nfev_box[0] += 1
            return 0

        # CVODE (via scikits.odes) accepts a per-state 1D array for
        # atol, matching the SUNDIALS interface. Pass it through
        # unchanged so each state component is controlled at its
        # physically correct tolerance. Collapsing to a scalar via
        # np.min (the previous behaviour) defeated the per-component
        # atol built in solve(), causing the tightest component's
        # atol to be applied uniformly to all other components.
        atol_cvode = (
            np.asarray(atol, dtype=float).ravel()
            if isinstance(atol, np.ndarray)
            else float(atol)
        )

        # Build the CVODE solver. lmm_type='BDF' matches SPIDER's
        # TSSundialsSetType(ts, SUNDIALS_BDF). Newton is CVODE's default
        # nonlinear solver; it uses the cached LU factorisation of the
        # iteration matrix I - gamma * J which is the critical
        # robustness advantage over scipy's simpler Newton.
        #
        # Linear solver choice: Aragog's Jacobian is pentadiagonal
        # (bandwidth 2) in quasi_steady mode because the flux divergence
        # operator only couples each staggered node to its 2 nearest
        # neighbours on each side. A dense FD Jacobian needs N=80 RHS
        # evals per Jacobian build; a banded FD Jacobian needs only
        # ~5 (bandwidth+1 per side). This is the CVODE equivalent of
        # the `jac_sparsity` graph-colouring hint scipy uses.
        #
        # Extended-state modes (energy_balance, gradient) break the
        # banded structure because the extra state row couples to
        # multiple entropy nodes; fall back to dense for those modes.
        cvode_options = {
            'old_api': False,
            'rtol': float(rtol),
            'atol': atol_cvode,
            'lmm_type': 'BDF',
            'nonlinsolver': 'newton',
            'max_steps': 100000,  # per-solve cap; scipy used unlimited
            # Maximum BDF order. CVODE's default is 5 (max of BDF(1..5))
            # but scikits.odes exposes the knob; set explicitly for
            # clarity. Matches SPIDER's SUNDIALS config.
            'order': 5,
            # Give Newton more iterations per step (default 3) and more
            # convergence-failure tolerance (default 10) before CVODE
            # demotes the order. Aragog's RHS has genuine physical
            # stiffness near the crystallisation front, and the default
            # counters trigger premature order-1 lock-in. Higher values
            # let CVODE iterate its way through the stiff transient
            # without permanently dropping order.
            'max_nonlin_iters': 10,
            'max_conv_fails': 25,
        }
        if self._core_bc == 'quasi_steady':
            cvode_options['linsolver'] = 'band'
            cvode_options['lband'] = 2
            cvode_options['uband'] = 2
        else:
            cvode_options['linsolver'] = 'dense'
        # max_step is only meaningful when < 1e100; otherwise CVODE
        # picks its own maximum internal step.
        if max_step is not None and np.isfinite(max_step):
            cvode_options['max_step_size'] = float(max_step)

        solver = _scikits_ode('cvode', rhs_fn, **cvode_options)
        # One-time debug print of the CVODE options actually in effect.
        # This helps verify banded linsolver dispatch vs silent fallback.
        if not hasattr(EntropySolver, '_cvode_options_logged'):
            EntropySolver._cvode_options_logged = True
            logger.info(
                'CVODE options in use: %s',
                {k: v for k, v in cvode_options.items() if k != 'old_api'},
            )
        cvode_sol = solver.solve(
            np.array([start_time, end_time], dtype=float),
            np.asarray(y0, dtype=float).ravel(),
        )

        # Dump CVODE's internal counters. These expose what scipy hides
        # from us and let us discriminate Newton thrash from step-size
        # control thrash from linear-solver thrash.
        #   NumSteps         = nst   (internal BDF steps taken)
        #   NumRhsEvals      = nfe   (RHS evaluations as counted by CVODE)
        #   NumLinSolvSetups = nsetups (iteration-matrix rebuilds, i.e.
        #                      Jacobian FD + LU factorisations)
        #   NumErrTestFails  = netf  (local truncation error test failures)
        #   LastOrder/CurrentOrder = BDF order actually in use (1..5)
        try:
            info = solver._integrator.get_info()
            logger.info(
                'CVODE stats: nst=%d nfe=%d nsetups=%d netf=%d '
                'order=%d last_step=%.2e cur_step=%.2e '
                'rhs_wrap=%d rhs_wrap/nst=%.1f nsetups/nst=%.2f',
                info.get('NumSteps', -1),
                info.get('NumRhsEvals', -1),
                info.get('NumLinSolvSetups', -1),
                info.get('NumErrTestFails', -1),
                info.get('CurrentOrder', info.get('LastOrder', -1)),
                info.get('LastStep', float('nan')),
                info.get('CurrentStep', float('nan')),
                nfev_box[0],
                nfev_box[0] / max(int(info.get('NumSteps', 1)), 1),
                int(info.get('NumLinSolvSetups', 0))
                    / max(int(info.get('NumSteps', 1)), 1),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug('CVODE get_info() failed: %s', exc)

        # Build a scipy-compatible result wrapper. scipy's
        # OptimizeResult attributes used downstream:
        #   .t      (1D array of internal time points)
        #   .y      (2D array, shape (n_state, n_time), final state at [:, -1])
        #   .status (0 = success, 1 = termination event, -1 = failure)
        #   .message (human-readable)
        #   .nfev   (RHS call count)
        result = OptimizeResult()
        if cvode_sol.values is not None and cvode_sol.values.t is not None:
            t_arr = np.asarray(cvode_sol.values.t, dtype=float).ravel()
            y_arr = np.asarray(cvode_sol.values.y, dtype=float)
            # CVODE stores y as (n_time, n_state); scipy expects
            # (n_state, n_time).
            if y_arr.ndim == 2:
                y_arr = y_arr.T
            else:
                y_arr = y_arr.reshape(-1, 1)
            result.t = t_arr
            result.y = y_arr
        else:
            result.t = np.array([start_time])
            result.y = np.asarray(y0, dtype=float).reshape(-1, 1)

        result.nfev = nfev_box[0]
        result.message = getattr(cvode_sol, 'message', '')
        # CVODE flag 0 = success, 2 = root found (tstop), negative = failure.
        flag = int(getattr(cvode_sol, 'flag', -1))
        if flag == 0 or flag == 2:
            result.status = 0
        else:
            result.status = -1
            if not result.message:
                result.message = f'CVODE failed with flag {flag}'
        return result

    def solve(self) -> None:
        """Run the BDF time integration."""
        start_time = self.parameters.solver.start_time
        end_time = self.parameters.solver.end_time
        # Test 2026-04-11 evening: lower the atol floor from 0.01 to
        # 1e-8 to match SPIDER's `atol = rtol = 1e-8` exactly. SPIDER
        # integrates with 8 orders of magnitude tighter absolute
        # tolerance; if cumulative integration error over ~200 kyr of
        # solidification contributes to the observed -18% T_core gap
        # vs SPIDER, tightening atol should shift the trajectory.
        atol_base = max(self.parameters.solver.atol, 1.0e-8)
        rtol = self.parameters.solver.rtol

        # Phase-aware atol: tight during crystallization, relaxed only
        # when fully solid (Fix B, 2026-04-10).
        try:
            n_stag = self._n_stag
            if self._core_bc == 'gradient':
                # Gradient mode: reconstruct S to compute phi0
                n_basic = n_stag + 1
                dSdr0 = self._S0[:n_basic]
                S_surf0 = float(self._S0[n_basic])
                S0_block, _ = self._reconstruct_entropy(dSdr0, S_surf0)
            elif self._state_is_extended:
                S0_block = self._S0[:n_stag]
            else:
                S0_block = self._S0
            phi0 = float(np.asarray(
                self.entropy_eos.melt_fraction(self._P_stag_flat, S0_block)
            ).mean())
            phi0 = max(0.0, min(1.0, phi0))
        except Exception:
            phi0 = 1.0

        if phi0 > 0.01:
            atol_scale = 1.0
            max_step = 100.0  # years
        else:
            atol_scale = 10.0
            max_step = np.inf

        # Tighten max_step when the CMB cell is near the liquidus.
        # Applies to ALL core_bc modes. For value-based modes, S[0]
        # is the direct state variable and atol controls its error,
        # so max_step=1 yr combined with the liquidus event gives
        # the BDF enough resolution to crystallize gradually.
        if phi0 > 0.01 and self.entropy_eos is not None:
            P_cmb = float(self._P_basic_flat[0])
            S_liq = float(self.entropy_eos.liquidus_entropy(
                np.array([P_cmb])).item())
            S_sol = float(self.entropy_eos.solidus_entropy(
                np.array([P_cmb])).item())
            S0_block_cmb = float(S0_block[0])
            margin = S0_block_cmb - S_liq
            if margin < 200.0 or (S0_block_cmb < S_liq and S0_block_cmb > S_sol):
                max_step = 1.0

        atol = atol_base * atol_scale

        logger.info(
            'EntropySolver: integrating from %.2e to %.2e yr '
            '(Phi_init=%.3f, atol_scale=%.1fx, atol=%.2e, rtol=%.2e)',
            start_time, end_time, phi0, atol_scale, atol, rtol,
        )

        # ── Nondimensionalise state, time, and tolerances ──
        # All state components become O(1) after dividing by their
        # reference scales, so scalar atol works uniformly and the
        # BDF solver no longer needs to resolve 9+ significant digits
        # on an O(1000) state variable. The physics code in
        # _dSdt_single stays in physical units; only the solver
        # interface scales in and out.
        S_ref = self._S_ref
        t_ref = self._t_ref_yr
        dSdr_ref = self._dSdr_ref
        n_s = self._n_stag

        # Per-component scale vector: y_phys = y_nd * _state_scale
        _state_scale = np.full(len(self._S0), S_ref)
        if self._core_bc == 'gradient':
            nb = n_s + 1
            _state_scale[:nb] = dSdr_ref   # dS/dr components
            _state_scale[nb] = S_ref        # S_surf
        elif self._state_is_extended:
            _state_scale[:n_s] = S_ref      # entropy
            _state_scale[n_s] = dSdr_ref    # dSdr_cmb (energy_balance)

        # Precompute RHS scale: dydt_nd = dydt_phys * _rhs_scale
        _rhs_scale = t_ref / _state_scale

        S0_nd = self._S0 / _state_scale
        start_nd = start_time / t_ref
        end_nd = end_time / t_ref
        max_step_nd = max_step / t_ref if np.isfinite(max_step) else max_step

        # Per-component atol in nondim units. Each state component
        # has its own nondim scale (S_ref for entropy, dSdr_ref for
        # dSdr_cmb in energy_balance mode, dSdr_ref for all gradient
        # components in gradient mode). Dividing atol by _state_scale
        # element-wise gives the same physical tolerance on every
        # component, regardless of its nondim scale.
        #
        # Previously this was scalar atol_nd = atol/S_ref, which
        # applied correctly to entropy but forced ~10^8x tighter
        # physical tolerance on dSdr_cmb. CVODE then took very small
        # steps to resolve dSdr_cmb, suppressing the entropy cooling
        # rate by a large factor and preventing solidification.
        atol_nd = atol / _state_scale

        def _rhs_nondim(t_nd, y_nd):
            """Nondim wrapper: scale state to physical, call physics, scale RHS back."""
            dydt_phys = self._dSdt_single(
                t_nd * t_ref, y_nd * _state_scale,
            )
            return dydt_phys * _rhs_scale

        logger.info(
            'Nondimensionalisation: S_ref=%.3f t_ref=%.3e yr '
            'atol_nd min/max=[%.2e, %.2e] S0_nd range [%.4f, %.4f]',
            S_ref, t_ref,
            float(np.min(atol_nd)), float(np.max(atol_nd)),
            S0_nd[:n_s].min(), S0_nd[:n_s].max(),
        )

        # BDF integration with phase-aware max_step constraint.
        jac_sparsity = self._build_jac_sparsity()

        # No terminal event. The max_step=1 yr near the liquidus
        # (set above) is sufficient for the value-based formulation
        # where atol directly controls S[0]. Terminal events cause
        # the solver to get stuck at the liquidus indefinitely.
        events = None

        # SOLVER: SUNDIALS CVODE via scikits.odes (2026-04-11). Same
        # solver SPIDER uses (CVODE BDF with modified-Newton). scipy's
        # BDF and Radau both hit `Required step size is less than
        # spacing between numbers` at the crystallisation front because
        # scipy's Newton iterator cannot converge on the stiff phase
        # transition. CVODE's C implementation handles it cleanly.
        # Scipy solve_ivp is kept as a fallback only if CVODE is not
        # available (e.g., scikits.odes import fails).
        solver_method = getattr(
            self.parameters.energy, 'solver_method', 'radau'
        )
        use_cvode = (
            solver_method == 'cvode'
            and _CVODE_AVAILABLE
        )
        if use_cvode:
            logger.info('EntropySolver: using CVODE (solver_method=cvode)')
            self._solution = self._solve_cvode(
                start_time=start_nd,
                end_time=end_nd,
                y0=S0_nd,
                atol=atol_nd,
                rtol=rtol,
                max_step=max_step_nd,
                rhs=_rhs_nondim,
            )
        else:
            method = 'Radau' if solver_method != 'bdf' else 'BDF'
            logger.info('EntropySolver: using scipy %s', method)
            self._solution = solve_ivp(
                _rhs_nondim,
                (start_nd, end_nd),
                S0_nd,
                method=method,
                vectorized=False,
                dense_output=False,
                atol=atol_nd,
                rtol=rtol,
                jac_sparsity=jac_sparsity,
                max_step=max_step_nd,
                events=events,
            )

        # ── Restore physical units ──
        sol = self._solution
        if sol.t is not None:
            sol.t = np.asarray(sol.t, dtype=float) * t_ref
        if sol.y is not None:
            sol_y = np.asarray(sol.y, dtype=float)
            if sol_y.ndim == 2:
                sol.y = sol_y * _state_scale[:, np.newaxis]
            else:
                sol.y = sol_y * _state_scale

        # Diagnostic logging: internal BDF step statistics (in physical yr)
        if sol.t is not None and len(sol.t) > 1:
            dt_internal = np.diff(sol.t)
            logger.info(
                'EntropySolver: %d internal steps, %d RHS evals, '
                'dt_min=%.2e yr, dt_max=%.2e yr, dt_med=%.2e yr',
                len(sol.t), sol.nfev,
                dt_internal.min(), dt_internal.max(),
                np.median(dt_internal),
            )
            logger.info(
                'EntropySolver: phase-boundary cache hits=%d misses=%d',
                self.state._pb_cache_hits,
                self.state._pb_cache_misses,
            )
            self.state._pb_cache_hits = 0
            self.state._pb_cache_misses = 0

        if self._solution.status == 0:
            logger.info('EntropySolver: integration completed successfully.')
            self.stop_early = False
        elif self._solution.status == 1:
            # Termination event (liquidus crossing at CMB cell).
            # Integration succeeded up to the event time.
            t_event = self._solution.t[-1]
            logger.info(
                'EntropySolver: liquidus-crossing event at t=%.2e yr '
                '(stopped %.1f yr before end_time). Bottom cell reached '
                'onset of crystallization.',
                t_event, end_time - t_event,
            )
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
        energy_balance = (self._core_bc == 'energy_balance')
        gradient_mode = (self._core_bc == 'gradient')
        is_ext = energy_balance

        # Slice the final state vector.
        if gradient_mode:
            n_basic = n_stag + 1
            dSdr_final = sol.y[:n_basic, -1]
            S_surf_final = float(sol.y[n_basic, -1])
            S_final, S_basic_final = self._reconstruct_entropy(
                dSdr_final, S_surf_final
            )
            extra_final = None
        elif is_ext:
            S_final = sol.y[:n_stag, -1]
            extra_final = float(sol.y[n_stag, -1])
        else:
            S_final = sol.y[:, -1]
            extra_final = None

        P_stag = self._P_stag_flat
        r_basic = self._r_basic_flat
        r_stag = np.asarray(mesh.staggered.radii).ravel()
        vol = self._volume_flat

        if eos is not None:
            T_stag = np.asarray(eos.temperature(P_stag, S_final)).ravel()
            phi_stag = np.asarray(eos.melt_fraction(P_stag, S_final)).ravel()
            rho_stag = np.asarray(eos.density(P_stag, S_final)).ravel()
        else:
            # const_properties: analytical T, phi=1, rho=const
            pm = self.parameters.phase_mixed
            T_stag = pm.const_T_ref * np.exp((S_final - pm.const_S_ref) / pm.const_Cp)
            phi_stag = np.ones_like(S_final)
            rho_stag = np.full_like(S_final, pm.const_rho)

        # Refresh the state at the final entropy for derived quantities.
        if gradient_mode:
            self.state.update(
                S_final, sol.t[-1], dSdr=dSdr_final, entropy_basic=S_basic_final
            )
        elif energy_balance:
            self.state.update(S_final, sol.t[-1], dSdr_cmb=extra_final)
        else:
            self.state.update(S_final, sol.t[-1])
        visc_stag = np.asarray(self.state.phase_staggered.viscosity()).ravel()
        heat_flux = self.state.heat_flux.copy()
        heating = self.state.heating.copy()
        eddy_diff = self.state.eddy_diffusivity.copy()
        cap_stag = np.asarray(self.state.capacitance_staggered()).ravel()

        # Diagnostic snapshot from the post-integration state.update().
        jcond_b = self.state.jcond.copy()
        jconv_b = self.state.jconv.copy()
        jgrav_b = self.state.jgrav_heat.copy()
        jmix_b = self.state.jmix_heat.copy()
        dSdr_b = self.state.dSdr.copy()
        phi_basic_diag = self.state.phi_basic_diag.copy()
        T_basic_diag = self.state.T_basic_diag.copy()
        cp_basic_diag = self.state.cp_basic_diag.copy()
        rho_basic_diag = self.state.rho_basic_diag.copy()

        # Scalar quantities.
        # M_mantle uses the analytic A-W mass integral (matching
        # SPIDER's EOSAdamsWilliamson_GetMassWithinShell) when
        # eos_method=1. The discrete sum (rho_stag * vol) has O(h^2)
        # quadrature error vs the analytic integral, causing the
        # structure root finder to converge to a different R_int.
        # For eos_method=2 (external mesh), fall back to discrete sum.
        mesh = self.evaluator.mesh
        if hasattr(mesh.eos, 'get_mass_within_radii'):
            r_cmb = float(self._r_basic_flat[0])
            r_surf = float(self._r_basic_flat[-1])
            M_mantle = (
                mesh.eos.get_mass_within_radii(np.array([r_surf]))
                - mesh.eos.get_mass_within_radii(np.array([r_cmb]))
            ).item()
        else:
            rho_struct_stag = np.asarray(
                mesh.staggered_effective_density
            ).ravel()
            mass_stag = rho_struct_stag * vol
            M_mantle = float(np.sum(mass_stag))
        mass_stag = rho_stag * vol  # PALEOS density for per-cell output
        # T_magma = top BASIC node T, SPIDER-parity with
        # `atmosphere/temperature_surface` in SPIDER's JSON output
        # (`interior_energetics/spider.py:1237`). The top basic node is at
        # r = outer_boundary where P = surface_pressure (= 0 for the
        # default Adams-Williamson BC); the top staggered cell is half a
        # cell below it at non-zero P. Previously we reported
        # `T_magma = T_stag[-1]`, which for the CHILI R8 config at
        # S=3900 J/kg/K, 1TPa-dK09 tables, rho_s=4000 kg/m^3 placed
        # T_magma at (P~2770 bar, S=3900) = 3862 K instead of
        # (P=0 bar, S=3900) = 3820 K. PROTEUS passes `hf_row['T_magma']`
        # to AGNI as the interior-side surface boundary condition
        # (`atmos_clim/agni.py:441`), so the ~44 K offset drove a
        # +12% F_atm offset at iteration 1, which cumulatively explained
        # the full -18% T_core endpoint gap against SPIDER v4.
        T_basic_final = np.asarray(
            self.state.phase_basic.temperature()
        ).ravel()
        T_magma = float(T_basic_final[-1])
        # Core temperature: bottom staggered cell (T_stag[0]).
        # SPIDER reports T_core = interior_o.temp[-1] which is the last
        # staggered node (SPIDER orders surface-to-CMB, so [-1] = CMB
        # cell). Aragog orders CMB-to-surface, so [0] = CMB cell.
        # Previous code used the CMB basic-node entropy via EOS for
        # energy_balance/gradient modes, but that is at the actual CMB
        # radius (half a cell below T_stag[0]), giving a systematic
        # +10 K offset from the higher pressure. Using T_stag[0] for
        # all modes matches SPIDER's definition exactly.
        T_core = float(T_stag[0])
        Phi_global = float(np.dot(phi_stag, vol) / np.sum(vol))

        # Rheological front depth. ``phi_basic_stag_interp`` is the
        # staggered-phi interpolated to basic nodes -- distinct from the
        # EOS-evaluated ``phi_basic_diag`` captured for the diagnostic
        # output above, which is ``phase_basic.melt_fraction()``.
        phi_rheo = self.parameters.phase_mixed.rheological_transition_melt_fraction
        phi_basic_stag_interp = mesh.quantity_at_basic_nodes(phi_stag).ravel()
        if Phi_global > 0.99:
            rf = float(r_basic[0])
        elif Phi_global < 0.01:
            rf = float(r_basic[-1])
        else:
            idx = np.argmin(np.abs(phi_basic_stag_interp - phi_rheo))
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
        if eos is not None:
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
        else:
            Phi_global_vol = 1.0  # const_properties: fully liquid

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
            jcond_b=jcond_b,
            jconv_b=jconv_b,
            jgrav_b=jgrav_b,
            jmix_b=jmix_b,
            dSdr_b=dSdr_b,
            phi_basic=phi_basic_diag,
            T_basic=T_basic_diag,
            cp_basic=cp_basic_diag,
            rho_basic=rho_basic_diag,
        )
