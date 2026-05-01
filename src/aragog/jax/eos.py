"""JAX-based entropy EOS layer for PALEOS P-S tables.

Drop-in replacement for aragog.eos.entropy.EntropyEOS using JAX arrays
and jax.scipy.interpolate.RegularGridInterpolator. All methods are
JIT-compilable and differentiable via jax.grad.

Table loading uses the existing numpy loader (disk I/O is not JIT-compiled).
The loaded grids are converted to JAX arrays and stored as equinox Module
fields so the entire EOS object is a valid JAX pytree.

Dependencies: jax, equinox (both already in PROTEUS ecosystem via atmodeller).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 (atmodeller does the same in its __init__.py)
jax.config.update('jax_enable_x64', True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPIDER-parity combined phase state (mirrors numpy EntropyPhaseEvaluator
# ._update_eos: single-pass evaluation, all properties derived from one
# shared (S_sol, S_liq, gphi, smth, T_sol, T_liq, rho_sol, rho_liq) cache).
# ---------------------------------------------------------------------------

class PhaseState(NamedTuple):
    """Material properties at (P, S) following SPIDER eos_composite.c convention.

    All scalar properties (T, rho, Cp, alpha, dTdPs, k) result from a
    smth-blend between two-phase ``mixed`` values (analytical formulas at
    the phase boundaries) and ``single`` table values (looked up at the
    actual P and at S_sol or S_liq when mushy). This matches numpy
    ``EntropyPhaseEvaluator._update_eos`` step-for-step.
    """

    temperature: jax.Array          # [K]
    density: jax.Array              # [kg/m^3]
    heat_capacity: jax.Array        # [J/kg/K]
    thermal_expansivity: jax.Array  # [1/K]
    dTdPs: jax.Array                # [K/Pa]
    thermal_conductivity: jax.Array # [W/m/K]
    melt_fraction: jax.Array        # phi, clipped to [0, 1]
    gphi: jax.Array                 # untruncated melt fraction
    smth: jax.Array                 # mixed-vs-single blend factor
    latent_heat: jax.Array          # [J/kg]


def _tanh_weight_jax(x: jax.Array, threshold: float, width: float) -> jax.Array:
    """0.5 * (1 + tanh((x - threshold) / width)). Mirrors aragog.utilities.tanh_weight."""
    return 0.5 * (1.0 + jnp.tanh((x - threshold) / width))


# ---------------------------------------------------------------------------
# Table loading (numpy, not JIT-compiled)
# ---------------------------------------------------------------------------

def _load_spider_ps_table(filepath: Path) -> dict:
    """Load a SPIDER-format P-S property table from disk.

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    dict
        Keys: 'P' (1D float64), 'S' (1D float64), 'values' (2D float64,
        shape (n_P, n_S)), 'n_P', 'n_S'.
    """
    with open(filepath) as f:
        header = f.readline().strip()
        parts = header.split()
        n_header = int(parts[1])
        n_P = int(parts[2])
        n_S = int(parts[3])
        for _ in range(n_header - 1):
            line = f.readline()
        scales_line = line.strip().lstrip('#').strip()
        scales = scales_line.split()
        P_scale = float(scales[0])
        S_scale = float(scales[1])
        Q_scale = float(scales[2])

    data = np.genfromtxt(filepath, skip_header=n_header)
    P_all = data[:, 0] * P_scale
    S_all = data[:, 1] * S_scale
    Q_all = data[:, 2] * Q_scale

    P_unique = np.unique(P_all)
    S_unique = np.unique(S_all)

    if len(P_unique) != n_P or len(S_unique) != n_S:
        logger.warning(
            'Table %s: expected %d x %d grid, got %d x %d unique values',
            filepath.name, n_P, n_S, len(P_unique), len(S_unique),
        )

    # SPIDER writes S varying slowest, P varying fastest.
    # Reshape to (n_S, n_P) then transpose to (n_P, n_S).
    values = Q_all.reshape(n_S, n_P).T

    return {
        'P': P_unique,
        'S': S_unique,
        'values': values,
        'n_P': len(P_unique),
        'n_S': len(S_unique),
    }


def _load_spider_phase_boundary(filepath: Path) -> dict:
    """Load a SPIDER-format phase boundary file (solidus or liquidus in P-S).

    Returns
    -------
    dict
        Keys: 'P' (1D float64), 'S' (1D float64).
    """
    with open(filepath) as f:
        header = f.readline().strip()
        parts = header.split()
        n_header = int(parts[1])
        for _ in range(n_header - 1):
            line = f.readline()
        scales_line = line.strip().lstrip('#').strip()
        scales = scales_line.split()
        P_scale = float(scales[0])
        S_scale = float(scales[1])

    data = np.genfromtxt(filepath, skip_header=n_header)
    P = data[:, 0] * P_scale
    S = data[:, 1] * S_scale

    return {'P': P, 'S': S}


# ---------------------------------------------------------------------------
# JAX interpolator helpers
# ---------------------------------------------------------------------------

def _bilinear_interp(
    P_grid: jax.Array, S_grid: jax.Array, values: jax.Array,
    P_query: jax.Array, S_query: jax.Array,
) -> jax.Array:
    """Bilinear interpolation on a regular (P, S) grid.

    Pure JAX implementation with no external interpolator objects.
    Clamps queries to the grid domain. Equivalent to scipy/jax
    RegularGridInterpolator with method='linear'.

    Parameters
    ----------
    P_grid : (n_P,) sorted ascending
    S_grid : (n_S,) sorted ascending
    values : (n_P, n_S) property values
    P_query, S_query : (N,) query points

    Returns
    -------
    (N,) interpolated values
    """
    # Clamp to grid domain
    P_c = jnp.clip(P_query, P_grid[0], P_grid[-1])
    S_c = jnp.clip(S_query, S_grid[0], S_grid[-1])

    # Find grid indices: searchsorted gives the index of the right edge.
    # Clamp to [0, n-2] so i and i+1 are both valid indices.
    #
    # stop_gradient on indices: they are discrete (non-differentiable)
    # and keeping them in the trace graph makes implicit solver JIT
    # compilation intractable. This gives the EXACT analytic Jacobian
    # of the bilinear interpolation WITHIN each grid cell, with
    # gradients flowing through tp, ts (smooth) and through the
    # constant cell-corner values v00, v10, v01, v11 (treated as
    # constant w.r.t. the query). At cell boundaries the bilinear
    # function is C^0 but its Jacobian is naturally discontinuous;
    # CVODE absorbs that as a stiff-region step-rejection event.
    # See test_jax_eos_jacobian_within_cell.py for the parity test.
    ip = jax.lax.stop_gradient(
        jnp.clip(jnp.searchsorted(P_grid, P_c, side='right') - 1, 0, len(P_grid) - 2)
    )
    js = jax.lax.stop_gradient(
        jnp.clip(jnp.searchsorted(S_grid, S_c, side='right') - 1, 0, len(S_grid) - 2)
    )

    # Fractional position within the cell
    P0 = P_grid[ip]
    P1 = P_grid[ip + 1]
    S0 = S_grid[js]
    S1 = S_grid[js + 1]

    tp = (P_c - P0) / jnp.maximum(P1 - P0, 1e-30)
    ts = (S_c - S0) / jnp.maximum(S1 - S0, 1e-30)

    # Four corner values
    v00 = values[ip, js]
    v10 = values[ip + 1, js]
    v01 = values[ip, js + 1]
    v11 = values[ip + 1, js + 1]

    # Bilinear blend
    return (
        v00 * (1 - tp) * (1 - ts)
        + v10 * tp * (1 - ts)
        + v01 * (1 - tp) * ts
        + v11 * tp * ts
    )


class _Table2D(eqx.Module):
    """A single 2D (P, S) property table as a JAX pytree.

    Stores the raw grid arrays and values. Interpolation is done
    with a pure JAX bilinear function (no scipy/jax
    RegularGridInterpolator object, which causes pytree issues
    with diffrax).
    """

    _P_grid: jax.Array
    _S_grid: jax.Array
    _values: jax.Array
    P_min: float
    P_max: float
    S_min: float
    S_max: float

    def __init__(self, P: np.ndarray, S: np.ndarray, values: np.ndarray):
        self._P_grid = jnp.asarray(P)
        self._S_grid = jnp.asarray(S)
        self._values = jnp.asarray(values)
        self.P_min = float(P[0])
        self.P_max = float(P[-1])
        self.S_min = float(S[0])
        self.S_max = float(S[-1])

    def __call__(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Query the table at (P, S), clamping to the table domain."""
        return _bilinear_interp(
            self._P_grid, self._S_grid, self._values, P, S,
        )


class _PhaseBoundary1D(eqx.Module):
    """A 1D phase boundary curve S(P) as a JAX-compatible pytree.

    Uses jnp.interp (linear, with end-value clamping). Also precomputes
    dS/dP via finite differences (np.gradient) so the SPIDER-parity
    bracket Jmix formula can evaluate dS_sol/dP and dS_liq/dP at basic
    nodes without rebuilding an interp1d each call. Mirrors the numpy
    ``_load_spider_phase_boundary`` dinterp path (entropy.py:279-296).
    """

    _P: jax.Array
    _S: jax.Array
    _dSdP: jax.Array

    def __init__(self, P: np.ndarray, S: np.ndarray):
        self._P = jnp.asarray(P)
        self._S = jnp.asarray(S)
        # np.gradient matches the numpy reference (entropy.py:279) exactly
        # so downstream dS/dP values are bit-identical across paths.
        self._dSdP = jnp.asarray(np.gradient(np.asarray(S), np.asarray(P)))

    def __call__(self, P: jax.Array) -> jax.Array:
        """Evaluate S_boundary(P) with linear interpolation and clamping."""
        return jnp.interp(P, self._P, self._S)

    def dSdP(self, P: jax.Array) -> jax.Array:
        """Evaluate dS_boundary/dP(P) [J/(kg·K·Pa)].

        Outside the tabulated range, ``jnp.interp`` returns the edge
        values (which are zero in practice since ``np.gradient`` uses
        one-sided differences at the ends). This matches the numpy
        dinterp clamping-to-zero behaviour (entropy.py:280-281).
        """
        return jnp.interp(P, self._P, self._dSdP)


# ---------------------------------------------------------------------------
# Main EOS class
# ---------------------------------------------------------------------------

class EntropyEOS_JAX(eqx.Module):
    """JAX-based entropy EOS from PALEOS P-S tables.

    Drop-in replacement for ``aragog.eos.entropy.EntropyEOS`` with all
    lookups JIT-compilable and differentiable. Constructed from the same
    SPIDER-format table files.

    Parameters
    ----------
    eos_dir : Path or str
        Directory containing the SPIDER-format P-S table files.
    """

    # Property tables (4 properties x 2 phases = 8 tables)
    _temperature_solid: _Table2D
    _temperature_melt: _Table2D
    _density_solid: _Table2D
    _density_melt: _Table2D
    _heat_capacity_solid: _Table2D
    _heat_capacity_melt: _Table2D
    _dTdPs_solid: _Table2D
    _dTdPs_melt: _Table2D

    # Phase boundaries
    _solidus: _PhaseBoundary1D
    _liquidus: _PhaseBoundary1D

    # Domain bounds (for external use)
    P_min: float
    P_max: float
    S_min: float
    S_max: float

    def __init__(self, eos_dir: Path | str):
        eos_dir = Path(eos_dir)
        if not eos_dir.is_dir():
            raise FileNotFoundError(f'EOS directory not found: {eos_dir}')

        logger.info('Loading JAX entropy EOS from %s', eos_dir)

        # Load tables from disk (numpy) and convert to JAX interpolators
        def _make_table(name: str, phase: str) -> _Table2D:
            if name == 'dTdPs':
                fname = f'adiabat_temp_grad_{phase}.dat'
            else:
                fname = f'{name}_{phase}.dat'
            t = _load_spider_ps_table(eos_dir / fname)
            return _Table2D(t['P'], t['S'], t['values'])

        self._temperature_solid = _make_table('temperature', 'solid')
        self._temperature_melt = _make_table('temperature', 'melt')
        self._density_solid = _make_table('density', 'solid')
        self._density_melt = _make_table('density', 'melt')
        self._heat_capacity_solid = _make_table('heat_capacity', 'solid')
        self._heat_capacity_melt = _make_table('heat_capacity', 'melt')
        self._dTdPs_solid = _make_table('dTdPs', 'solid')
        self._dTdPs_melt = _make_table('dTdPs', 'melt')

        # Phase boundaries
        sol = _load_spider_phase_boundary(eos_dir / 'solidus_P-S.dat')
        liq = _load_spider_phase_boundary(eos_dir / 'liquidus_P-S.dat')
        self._solidus = _PhaseBoundary1D(sol['P'], sol['S'])
        self._liquidus = _PhaseBoundary1D(liq['P'], liq['S'])

        # Domain bounds
        self.P_min = min(self._temperature_solid.P_min, self._temperature_melt.P_min)
        self.P_max = max(self._temperature_solid.P_max, self._temperature_melt.P_max)
        self.S_min = min(self._temperature_solid.S_min, self._temperature_melt.S_min)
        self.S_max = max(self._temperature_solid.S_max, self._temperature_melt.S_max)

        logger.info(
            'JAX EOS loaded: P=[%.2e, %.2e] Pa, S=[%.0f, %.0f] J/kg/K',
            self.P_min, self.P_max, self.S_min, self.S_max,
        )

    # ------------------------------------------------------------------
    # Phase boundaries
    # ------------------------------------------------------------------

    def solidus_entropy(self, P: jax.Array) -> jax.Array:
        """Solidus entropy S_sol(P) [J/kg/K]."""
        return self._solidus(P)

    def liquidus_entropy(self, P: jax.Array) -> jax.Array:
        """Liquidus entropy S_liq(P) [J/kg/K]."""
        return self._liquidus(P)

    def solidus_entropy_dP(self, P: jax.Array) -> jax.Array:
        """dS_sol/dP at the given pressure(s), in J/(kg·K·Pa).

        Needed by the SPIDER-parity bracket Jmix in
        ``aragog.jax.phase.compute_fluxes``. Mirrors numpy
        ``EntropyEOS.solidus_entropy_dP`` (entropy.py:377-388).
        """
        return self._solidus.dSdP(P)

    def liquidus_entropy_dP(self, P: jax.Array) -> jax.Array:
        """dS_liq/dP at the given pressure(s), in J/(kg·K·Pa)."""
        return self._liquidus.dSdP(P)

    def melt_fraction(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Melt fraction phi from position between solidus and liquidus.

        phi = 0 for S <= S_sol, phi = 1 for S >= S_liq, linear between.
        """
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)
        dS = jnp.maximum(S_liq - S_sol, 1e-10)
        return jnp.clip((S - S_sol) / dS, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal lookup helpers
    # ------------------------------------------------------------------

    def _get_tables(self, prop_name: str) -> tuple[_Table2D, _Table2D]:
        """Return (solid_table, melt_table) for a property name."""
        solid = getattr(self, f'_{prop_name}_solid')
        melt = getattr(self, f'_{prop_name}_melt')
        return solid, melt

    def _lookup_phase_weighted(
        self, prop_name: str, P: jax.Array, S: jax.Array,
    ) -> jax.Array:
        """Look up a property with phase weighting, matching numpy EntropyEOS.

        Mirrors ``aragog.eos.entropy.EntropyEOS._lookup_phase_weighted``:

        - Mushy zone (0 < phi < 1): evaluate the solid table at the
          solidus entropy S_sol(P) and the melt table at the liquidus
          entropy S_liq(P) (Lever Rule / phase-boundary end-members),
          then blend by phi.
        - Pure phase (phi = 0 or phi = 1): evaluate the active table at
          the actual S (clamped by the table itself).

        The earlier JAX implementation always evaluated both tables at
        the actual (P, S), which diverged from numpy by up to ~6 % in
        T, ~2 % in Cp, and ~15 % in dTdPs inside the mushy band.
        """
        phi = self.melt_fraction(P, S)
        solid_table, melt_table = self._get_tables(prop_name)

        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)

        mushy = (phi > 0) & (phi < 1)
        S_for_solid = jnp.where(mushy, S_sol, S)
        S_for_melt = jnp.where(mushy, S_liq, S)

        val_solid = solid_table(P, S_for_solid)
        val_melt = melt_table(P, S_for_melt)

        result = (
            jnp.where(phi > 0, phi * val_melt, 0.0)
            + jnp.where(phi < 1, (1.0 - phi) * val_solid, 0.0)
        )
        return result

    def _lookup_at_phase_boundary(
        self, prop_name: str, P: jax.Array, phase: str,
    ) -> jax.Array:
        """Look up a property at the solidus or liquidus for the given phase.

        Evaluates the solid table at S_solidus(P) or the melt table at
        S_liquidus(P), matching SPIDER's end-member approach.
        """
        solid_table, melt_table = self._get_tables(prop_name)
        if phase == 'solid':
            table = solid_table
            S_boundary = self.solidus_entropy(P)
        else:
            table = melt_table
            S_boundary = self.liquidus_entropy(P)
        return table(P, S_boundary)

    # ------------------------------------------------------------------
    # Public property lookups
    # ------------------------------------------------------------------

    def temperature(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Temperature T(P, S) [K]."""
        return self._lookup_phase_weighted('temperature', P, S)

    def density(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Density rho(P, S) [kg/m^3], matching numpy EntropyEOS.density.

        - Mushy zone (0 < phi < 1): harmonic mean of end-member
          densities evaluated at phase-boundary entropies (Lever Rule,
          SPIDER ``eos_composite.c:236-237``).
        - Pure phase (phi = 0 or phi = 1): evaluate the active single-
          phase table at the actual S (clamped by the table itself).
          SPIDER ``combine_matprop(smth=0, mixed, single)`` selects the
          single-phase branch in this regime; the earlier JAX
          implementation used the harmonic-mean form unconditionally,
          producing up to 21 % density bias in the fully-molten regime
          relative to numpy.
        """
        phi = self.melt_fraction(P, S)
        mushy = (phi > 0) & (phi < 1)

        solid_table, melt_table = self._get_tables('density')

        # Mushy zone: harmonic mean at phase boundaries
        rho_sol_boundary = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq_boundary = self._lookup_at_phase_boundary('density', P, 'melt')
        inv_rho_mushy = (
            phi / jnp.maximum(rho_liq_boundary, 1.0)
            + (1.0 - phi) / jnp.maximum(rho_sol_boundary, 1.0)
        )
        rho_mushy = 1.0 / jnp.maximum(inv_rho_mushy, 1e-30)

        # Single-phase: evaluate at actual S (clamped by the table). Pick
        # the melt table for phi >= 0.5, solid otherwise (matches numpy).
        rho_solid_single = solid_table(P, S)
        rho_melt_single = melt_table(P, S)
        rho_single = jnp.where(phi >= 0.5, rho_melt_single, rho_solid_single)

        return jnp.where(mushy, rho_mushy, rho_single)

    def heat_capacity(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Specific heat capacity Cp(P, S) [J/kg/K]."""
        return self._lookup_phase_weighted('heat_capacity', P, S)

    def dTdPs(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Adiabatic temperature gradient dT/dP|_S (P, S) [K/Pa]."""
        return self._lookup_phase_weighted('dTdPs', P, S)

    def latent_heat(self, P: jax.Array) -> jax.Array:
        """Latent heat L(P) = T_fus x (S_liq - S_sol) [J/kg]."""
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)
        T_sol = self._lookup_at_phase_boundary('temperature', P, 'solid')
        T_liq = self._lookup_at_phase_boundary('temperature', P, 'melt')
        T_fus = 0.5 * (T_sol + T_liq)
        return T_fus * jnp.maximum(S_liq - S_sol, 1.0)

    def thermal_expansivity(self, P: jax.Array, S: jax.Array) -> jax.Array:
        """Thermal expansivity alpha(P, S) [1/K].

        Derived: alpha = rho * Cp * |dTdPs| / T.
        """
        T = self.temperature(P, S)
        rho = self.density(P, S)
        Cp = self.heat_capacity(P, S)
        dTdPs_val = self.dTdPs(P, S)
        return rho * Cp * jnp.abs(dTdPs_val) / jnp.maximum(T, 1.0)

    # ------------------------------------------------------------------
    # SPIDER-parity combined evaluation
    # ------------------------------------------------------------------

    def compute_phase_state(
        self,
        P: jax.Array,
        S: jax.Array,
        k_solid: float,
        k_liquid: float,
        matprop_smooth_width: float = 0.0,
    ) -> PhaseState:
        """Single-pass SPIDER-parity phase evaluation (cp_blend='latent').

        Bit-for-bit mirror of numpy ``EntropyPhaseEvaluator._update_eos``
        with ``cp_blend='latent'``. All properties share the same
        intermediates, and each is the smth-blend
        ``smth * mixed + (1 - smth) * single`` matching SPIDER
        ``combine_matprop`` (eos_composite.c:278-285).

        Parameters
        ----------
        P, S : jax.Array
            Pressure [Pa] and entropy [J/kg/K], same shape.
        k_solid, k_liquid : float
            Single-phase thermal conductivities [W/m/K].
        matprop_smooth_width : float, default 0.0
            SPIDER's ``-matprop_smooth_width``. ``0.0`` reproduces the
            sharp ``smth=1`` inside [0,1] convention; ``0.01`` is the
            CHILI Earth production setting.

        Returns
        -------
        PhaseState
            All blended properties and shared intermediates.
        """
        # ── Step 1: phase boundaries (computed ONCE) ────────────────
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)
        dS_phase = jnp.maximum(S_liq - S_sol, 1e-10)
        gphi = (S - S_sol) / dS_phase
        phi_arr = jnp.clip(gphi, 0.0, 1.0)

        # smth: matprop_smooth_width blend factor
        # (SPIDER util.c:get_smoothing). matprop_smooth_width is a static
        # Python float here, so the if-branch is resolved at trace time.
        if matprop_smooth_width > 0:
            smth = jnp.where(
                gphi > 0.5,
                1.0 - _tanh_weight_jax(gphi, 1.0, matprop_smooth_width),
                _tanh_weight_jax(gphi, 0.0, matprop_smooth_width),
            )
        else:
            smth = jnp.where((gphi >= 0.0) & (gphi <= 1.0), 1.0, 0.0)

        # ── Step 2: phase-boundary table evaluations (ONCE each) ────
        T_sol = self._lookup_at_phase_boundary('temperature', P, 'solid')
        T_liq = self._lookup_at_phase_boundary('temperature', P, 'melt')
        rho_sol = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq = self._lookup_at_phase_boundary('density', P, 'melt')

        # ── Step 3: intermediate two-phase ('mixed') properties ─────
        dT_phase = jnp.maximum(T_liq - T_sol, 1e-10)
        T_avg = T_sol + 0.5 * dT_phase

        # T: linear blend
        T_mixed = phi_arr * T_liq + (1.0 - phi_arr) * T_sol

        # rho: harmonic mean
        inv_rho_mixed = (
            phi_arr / jnp.maximum(rho_liq, 1.0)
            + (1.0 - phi_arr) / jnp.maximum(rho_sol, 1.0)
        )
        rho_mixed = 1.0 / jnp.maximum(inv_rho_mixed, 1e-30)

        # alpha and Cp: latent-heat-augmented (SPIDER eos_composite.c:227-246)
        alpha_mixed = (rho_sol - rho_liq) / dT_phase / jnp.maximum(rho_mixed, 1.0)
        Cp_mixed = jnp.maximum((S_liq - S_sol) / dT_phase * T_avg, 100.0)

        # dTdPs: analytical from intermediates
        dTdPs_mixed = alpha_mixed * T_mixed / (
            jnp.maximum(rho_mixed, 1.0) * jnp.maximum(Cp_mixed, 100.0)
        )

        # cond: linear blend
        cond_mixed = phi_arr * k_liquid + (1.0 - phi_arr) * k_solid

        # ── Step 4: single-phase table evaluations (SPIDER 269-276) ──
        # Evaluate at S_sol/S_liq when mushy, at actual S otherwise.
        mushy = (phi_arr > 0) & (phi_arr < 1)
        S_for_solid = jnp.where(mushy, S_sol, S)
        S_for_melt = jnp.where(mushy, S_liq, S)

        def _table_lookup_blend(prop_name: str) -> jax.Array:
            solid_tbl, melt_tbl = self._get_tables(prop_name)
            v_sol = solid_tbl(P, S_for_solid)
            v_mel = melt_tbl(P, S_for_melt)
            return jnp.where(gphi > 0.5, v_mel, v_sol)

        T_single = _table_lookup_blend('temperature')
        rho_single = _table_lookup_blend('density')
        Cp_single = _table_lookup_blend('heat_capacity')
        dTdPs_single = _table_lookup_blend('dTdPs')
        # alpha derived from thermodynamic identity (no thermal_exp tables yet)
        alpha_single = (
            dTdPs_single * rho_single * Cp_single
            / jnp.maximum(T_single, 1.0)
        )
        cond_single = jnp.where(gphi > 0.5, k_liquid, k_solid)

        # ── Step 5: combine_matprop blend (SPIDER 278-285) ──────────
        def _blend(mixed, single):
            return smth * mixed + (1.0 - smth) * single

        temperature = _blend(T_mixed, T_single)
        density = _blend(rho_mixed, rho_single)
        heat_capacity = _blend(Cp_mixed, Cp_single)
        alpha_raw = _blend(alpha_mixed, alpha_single)
        dTdPs_val = _blend(dTdPs_mixed, dTdPs_single)
        thermal_conductivity = _blend(cond_mixed, cond_single)

        # Guard: clamp negative alpha (matches numpy line 309)
        eps_a = 1.0e-8
        thermal_expansivity = 0.5 * (
            alpha_raw + jnp.sqrt(alpha_raw * alpha_raw + eps_a * eps_a)
        )

        latent_heat = self.latent_heat(P)

        return PhaseState(
            temperature=temperature,
            density=density,
            heat_capacity=heat_capacity,
            thermal_expansivity=thermal_expansivity,
            dTdPs=dTdPs_val,
            thermal_conductivity=thermal_conductivity,
            melt_fraction=phi_arr,
            gphi=gphi,
            smth=smth,
            latent_heat=latent_heat,
        )
