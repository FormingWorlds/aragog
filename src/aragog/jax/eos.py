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

logger = logging.getLogger('fwl.' + __name__)

# Per-context occurrence counts for the out-of-range entropy warning below.
# Keyed by the ``context`` string passed to ``_check_entropy_range``, a
# small fixed set of call-site labels, not per-cell or per-value data.
_RANGE_WARNING_COUNTS: dict[str, int] = {}
_RANGE_WARNING_LOG_EVERY = 100


def reset_range_warning_counts() -> None:
    """Clear the per-context occurrence counts for the entropy range warning.

    The counts are process-lifetime state, not owned by any
    ``EntropyEOS_JAX`` instance or solve. Nothing in aragog calls this
    function on its own: the JAX CVODE factory that builds each
    ``EntropyEOS_JAX`` instance (registered externally through
    ``EntropySolver.set_jax_cvode_factory``) is responsible for calling
    it at the start of a solve, if that solve should log its own first
    out-of-range occurrence rather than inherit a count left over from
    an earlier solve in the same process. Test suites that reuse the
    same context strings across independent tests must call it too.
    """
    _RANGE_WARNING_COUNTS.clear()


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

    temperature: jax.Array  # [K]
    density: jax.Array  # [kg/m^3]
    heat_capacity: jax.Array  # [J/kg/K]
    thermal_expansivity: jax.Array  # [1/K]
    dTdPs: jax.Array  # [K/Pa]
    thermal_conductivity: jax.Array  # [W/m/K]
    melt_fraction: jax.Array  # phi, clipped to [0, 1]
    gphi: jax.Array  # untruncated melt fraction
    smth: jax.Array  # mixed-vs-single blend factor
    latent_heat: jax.Array  # [J/kg]


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
            filepath.name,
            n_P,
            n_S,
            len(P_unique),
            len(S_unique),
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
    P_grid: jax.Array,
    S_grid: jax.Array,
    values: jax.Array,
    P_query: jax.Array,
    S_query: jax.Array,
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
    return v00 * (1 - tp) * (1 - ts) + v10 * tp * (1 - ts) + v01 * (1 - tp) * ts + v11 * tp * ts


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
            self._P_grid,
            self._S_grid,
            self._values,
            P,
            S,
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
    strict_range : bool, default False
        Raise ``RuntimeError`` instead of only warning when a lookup
        entropy is non-finite or outside the table range. Set by whichever
        code constructs this class; aragog's own ``Parameters`` schema has
        no field for it, and no code path in aragog itself sets it. A
        production PROTEUS run gets the default (warn-only) unless the
        external JAX CVODE factory that builds this instance passes
        ``strict_range=True`` explicitly. Useful for tests and interactive
        debugging.
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

    strict_range: bool

    def __init__(self, eos_dir: Path | str, strict_range: bool = False):
        self.strict_range = strict_range
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

        # Domain bounds. P uses the temperature tables only (grid shape
        # reference for callers). S is the union across every loaded
        # table, so a value outside [S_min, S_max] is guaranteed to be
        # clamped by at least one table lookup below.
        self.P_min = min(self._temperature_solid.P_min, self._temperature_melt.P_min)
        self.P_max = max(self._temperature_solid.P_max, self._temperature_melt.P_max)
        self.S_min = min(
            self._temperature_solid.S_min,
            self._temperature_melt.S_min,
            self._density_solid.S_min,
            self._density_melt.S_min,
            self._heat_capacity_solid.S_min,
            self._heat_capacity_melt.S_min,
            self._dTdPs_solid.S_min,
            self._dTdPs_melt.S_min,
        )
        self.S_max = max(
            self._temperature_solid.S_max,
            self._temperature_melt.S_max,
            self._density_solid.S_max,
            self._density_melt.S_max,
            self._heat_capacity_solid.S_max,
            self._heat_capacity_melt.S_max,
            self._dTdPs_solid.S_max,
            self._dTdPs_melt.S_max,
        )

        logger.info(
            'JAX EOS loaded: P=[%.2e, %.2e] Pa, S=[%.0f, %.0f] J/kg/K',
            self.P_min,
            self.P_max,
            self.S_min,
            self.S_max,
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

    def _check_entropy_range(
        self,
        S: jax.Array,
        S_min: float,
        S_max: float,
        context: str,
    ) -> None:
        """Flag entropy that a table-edge clamp would otherwise hide.

        JIT-safe mirror of ``EntropyEOS._check_entropy_range``
        (aragog.eos.entropy) using ``jax.debug.callback``: a non-finite
        or far-out-of-range S still produces a finite property value
        once it is clamped to the table edge. Warn always, and raise
        when ``self.strict_range`` is set, so this does not silently
        pass as a valid table-edge value.

        The host callback only fires when there is something to report:
        it sits behind a ``lax.cond`` on the in-range path, since a host
        round trip on every call is the dominant cost of this check on
        the CVODE right-hand side and every step is in range once the
        solve is past its initial transient.

        Once triggered, the warning itself is rate-limited per call site
        (``context``): logged on the first occurrence and then every
        ``_RANGE_WARNING_LOG_EVERY`` occurrences after, so a solve stuck
        with persistently out-of-range entropy logs a bounded number of
        lines instead of one per RHS evaluation. A ``strict_range`` raise
        is unaffected by the throttle: it still fires on every occurrence.
        """
        non_finite = jnp.sum(~jnp.isfinite(S))
        out_of_range = jnp.sum(jnp.isfinite(S) & ((S < S_min) | (S > S_max)))
        has_issue = (non_finite > 0) | (out_of_range > 0)

        def _report(n_non_finite: jax.Array, n_out_of_range: jax.Array) -> None:
            n_non_finite = int(n_non_finite)
            n_out_of_range = int(n_out_of_range)
            count = _RANGE_WARNING_COUNTS.get(context, 0) + 1
            _RANGE_WARNING_COUNTS[context] = count
            if count == 1 or count % _RANGE_WARNING_LOG_EVERY == 0:
                logger.warning(
                    '%s: %d non-finite and %d out-of-range entropy value(s) '
                    '(table range [%.6g, %.6g]), occurrence %d',
                    context,
                    n_non_finite,
                    n_out_of_range,
                    S_min,
                    S_max,
                    count,
                )
            if self.strict_range:
                raise RuntimeError(
                    f'{context}: {n_non_finite} non-finite and {n_out_of_range} '
                    f'out-of-range entropy value(s) outside table range '
                    f'[{S_min:.6g}, {S_max:.6g}]'
                )

        def _report_branch(n_non_finite: jax.Array, n_out_of_range: jax.Array) -> None:
            jax.debug.callback(_report, n_non_finite, n_out_of_range)

        def _no_op_branch(n_non_finite: jax.Array, n_out_of_range: jax.Array) -> None:
            del n_non_finite, n_out_of_range

        jax.lax.cond(has_issue, _report_branch, _no_op_branch, non_finite, out_of_range)

    def _lookup_phase_weighted(
        self,
        prop_name: str,
        P: jax.Array,
        S: jax.Array,
    ) -> jax.Array:
        """Look up a property with phase weighting, matching numpy EntropyEOS.

        Mirrors ``aragog.eos.entropy.EntropyEOS._lookup_phase_weighted``:

        - Mushy zone (0 < phi < 1): evaluate the solid table at the
          solidus entropy S_sol(P) and the melt table at the liquidus
          entropy S_liq(P) (Lever Rule / phase-boundary end-members),
          then blend by phi.
        - Pure phase (phi = 0 or phi = 1): evaluate the active table at
          the actual S (clamped by the table itself).

        Evaluating both tables at the actual (P, S) inside the mushy
        band would produce values that are not on either phase
        boundary, diverging from the numpy EntropyEOS reference.
        """
        phi = self.melt_fraction(P, S)
        solid_table, melt_table = self._get_tables(prop_name)

        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)

        mushy = (phi > 0) & (phi < 1)
        S_for_solid = jnp.where(mushy, S_sol, S)
        S_for_melt = jnp.where(mushy, S_liq, S)

        # Only check a branch's S where it actually has nonzero weight in
        # the blend below (phi < 1 for solid, phi > 0 for melt). Written
        # as a negated >=/<= so a NaN phi (comparisons always False)
        # still passes through the check instead of being masked.
        solid_used = ~(phi >= 1)
        melt_used = ~(phi <= 0)
        self._check_entropy_range(
            jnp.where(solid_used, S_for_solid, solid_table.S_min),
            solid_table.S_min,
            solid_table.S_max,
            f'{prop_name} (solid table lookup)',
        )
        self._check_entropy_range(
            jnp.where(melt_used, S_for_melt, melt_table.S_min),
            melt_table.S_min,
            melt_table.S_max,
            f'{prop_name} (melt table lookup)',
        )

        val_solid = solid_table(P, S_for_solid)
        val_melt = melt_table(P, S_for_melt)

        result = jnp.where(phi > 0, phi * val_melt, 0.0) + jnp.where(
            phi < 1, (1.0 - phi) * val_solid, 0.0
        )
        # NaN phi (NaN S) makes both comparisons above False, which would
        # otherwise mask a NaN input into a false 0.0 result.
        return jnp.where(jnp.isnan(phi), jnp.nan, result)

    def _lookup_at_phase_boundary(
        self,
        prop_name: str,
        P: jax.Array,
        phase: str,
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
        self._check_entropy_range(
            S_boundary,
            table.S_min,
            table.S_max,
            f'{prop_name} (phase-boundary {phase} table lookup)',
        )
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
          single-phase branch in this regime. Using the harmonic-mean
          form unconditionally biases the fully-molten density
          relative to the numpy EntropyEOS.
        """
        phi = self.melt_fraction(P, S)
        mushy = (phi > 0) & (phi < 1)

        solid_table, melt_table = self._get_tables('density')

        # Mushy zone: harmonic mean at phase boundaries
        rho_sol_boundary = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq_boundary = self._lookup_at_phase_boundary('density', P, 'melt')
        inv_rho_mushy = phi / jnp.maximum(rho_liq_boundary, 1.0) + (1.0 - phi) / jnp.maximum(
            rho_sol_boundary, 1.0
        )
        rho_mushy = 1.0 / jnp.maximum(inv_rho_mushy, 1e-30)

        # Single-phase: evaluate at actual S (clamped by the table). Pick
        # the melt table for phi >= 0.5, solid otherwise (matches numpy).
        # NOTE: this 0.5 is a binary table-selector for the non-mushy
        # fallback only — outside the mushy band phi is essentially 0 or
        # 1 by construction. It is NOT the rheological critical melt
        # fraction (RCMF). The RCMF lives in
        # ``EntropyPhaseEvaluator._phi_rheo`` / ``PhaseParams.phi_rheo``
        # and drives the viscosity tanh blend separately.
        # Single-phase branch check: mask each side to the points where
        # that branch is actually used (mushy points use rho_mushy
        # instead, and phi >= 0.5 picks melt over solid), to avoid a
        # false positive on the discarded branch.
        melt_selected = phi >= 0.5
        solid_used = ~mushy & ~melt_selected
        melt_used = ~mushy & melt_selected
        self._check_entropy_range(
            jnp.where(solid_used, S, solid_table.S_min),
            solid_table.S_min,
            solid_table.S_max,
            'density (solid table lookup)',
        )
        self._check_entropy_range(
            jnp.where(melt_used, S, melt_table.S_min),
            melt_table.S_min,
            melt_table.S_max,
            'density (melt table lookup)',
        )

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
            production setting.

        Returns
        -------
        PhaseState
            All blended properties and shared intermediates.
        """
        # Root-cause check: catch entropy outside the EOS table domain
        # here, before any table lookup, mirroring numpy
        # ``EntropyPhaseEvaluator._update_eos`` (entropy_phase.py:210).
        self._check_entropy_range(S, self.S_min, self.S_max, 'entropy_phase (composite domain)')

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
        inv_rho_mixed = phi_arr / jnp.maximum(rho_liq, 1.0) + (1.0 - phi_arr) / jnp.maximum(
            rho_sol, 1.0
        )
        rho_mixed = 1.0 / jnp.maximum(inv_rho_mixed, 1e-30)

        # alpha and Cp: latent-heat-augmented (SPIDER eos_composite.c:227-246).
        # The 100 J/kg/K floor on Cp_mixed is a defensive guard against
        # division-by-near-zero when the latent budget collapses
        # (dT_phase very large, or S_liq -> S_sol, or T_avg small near
        # the eutectic). MgSiO3 production runs are always well above
        # this floor; a triggering EOS is the signal that the upstream
        # property tables need clipping, not Aragog's runtime.
        alpha_mixed = (rho_sol - rho_liq) / dT_phase / jnp.maximum(rho_mixed, 1.0)
        Cp_mixed = jnp.maximum((S_liq - S_sol) / dT_phase * T_avg, 100.0)

        # dTdPs: analytical from intermediates
        dTdPs_mixed = (
            alpha_mixed * T_mixed / (jnp.maximum(rho_mixed, 1.0) * jnp.maximum(Cp_mixed, 100.0))
        )

        # cond: linear blend
        cond_mixed = phi_arr * k_liquid + (1.0 - phi_arr) * k_solid

        # ── Step 4: single-phase table evaluations (SPIDER 269-276) ──
        # Evaluate at S_sol/S_liq when mushy, at actual S otherwise.
        mushy = (phi_arr > 0) & (phi_arr < 1)
        S_for_solid = jnp.where(mushy, S_sol, S)
        S_for_melt = jnp.where(mushy, S_liq, S)

        # The ``gphi > 0.5`` switches below pick which pure-phase table to
        # evaluate when the cell is *outside* the mushy mask. This 0.5 is
        # a binary discriminator (gphi outside [0,1] is by definition
        # super-liquidus or sub-solidus), not the rheological critical
        # melt fraction; see ``PhaseParams.phi_rheo`` for the RCMF.
        def _table_lookup_blend(prop_name: str) -> jax.Array:
            solid_tbl, melt_tbl = self._get_tables(prop_name)
            solid_used = ~(gphi > 0.5)
            melt_used = gphi > 0.5
            self._check_entropy_range(
                jnp.where(solid_used, S_for_solid, solid_tbl.S_min),
                solid_tbl.S_min,
                solid_tbl.S_max,
                f'{prop_name} (single-phase solid table lookup)',
            )
            self._check_entropy_range(
                jnp.where(melt_used, S_for_melt, melt_tbl.S_min),
                melt_tbl.S_min,
                melt_tbl.S_max,
                f'{prop_name} (single-phase melt table lookup)',
            )
            v_sol = solid_tbl(P, S_for_solid)
            v_mel = melt_tbl(P, S_for_melt)
            return jnp.where(gphi > 0.5, v_mel, v_sol)

        T_single = _table_lookup_blend('temperature')
        rho_single = _table_lookup_blend('density')
        Cp_single = _table_lookup_blend('heat_capacity')
        dTdPs_single = _table_lookup_blend('dTdPs')
        # alpha derived from thermodynamic identity (no thermal_exp tables yet)
        alpha_single = dTdPs_single * rho_single * Cp_single / jnp.maximum(T_single, 1.0)
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

        # Unconditional NaN backstop, independent of self.strict_range:
        # mirrors numpy's pre-existing downstream guard at the end of
        # EntropyPhaseEvaluator._update_eos (entropy_phase.py:384-400),
        # which always raises on NaN regardless of strict_range. Keeps
        # this backend's fault-handling contract identical to numpy's
        # for entropy that produces NaN after the composite-domain
        # check has already warned (or raised, under strict_range).
        #
        # The raise always fires when temperature has a NaN; only the
        # host round trip is skipped on the clean path, since a
        # jax.debug.callback dispatch is the dominant cost of this
        # check on the CVODE right-hand side (same reasoning as
        # ``_check_entropy_range`` above).
        def _raise_on_nan(temperature: np.ndarray, S: np.ndarray) -> None:
            n_nan = int(np.sum(np.isnan(temperature)))
            logger.error(
                'NaN from EOS lookup at %d nodes. S range: [%.0f, %.0f], '
                'table domain: [%.0f, %.0f] J/kg/K',
                n_nan,
                float(np.nanmin(S)),
                float(np.nanmax(S)),
                self.S_min,
                self.S_max,
            )
            raise RuntimeError(
                f'Entropy out of EOS table domain at {n_nan} nodes. '
                f'S range [{np.nanmin(S):.0f}, {np.nanmax(S):.0f}] vs '
                f'table [{self.S_min:.0f}, {self.S_max:.0f}]'
            )

        def _raise_branch(temperature: jax.Array, S: jax.Array) -> None:
            jax.debug.callback(_raise_on_nan, temperature, S)

        def _no_op_nan_branch(temperature: jax.Array, S: jax.Array) -> None:
            del temperature, S

        has_nan = jnp.any(~jnp.isfinite(temperature))
        jax.lax.cond(has_nan, _raise_branch, _no_op_nan_branch, temperature, S)

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
