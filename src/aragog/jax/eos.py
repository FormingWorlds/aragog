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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 (atmodeller does the same in its __init__.py)
jax.config.update('jax_enable_x64', True)

logger = logging.getLogger(__name__)


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
    # stop_gradient on indices: they are discrete (non-differentiable)
    # and keeping them in the trace graph makes implicit solver JIT
    # compilation intractable.
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

    Uses jnp.interp (linear, with end-value clamping).
    """

    _P: jax.Array
    _S: jax.Array

    def __init__(self, P: np.ndarray, S: np.ndarray):
        self._P = jnp.asarray(P)
        self._S = jnp.asarray(S)

    def __call__(self, P: jax.Array) -> jax.Array:
        """Evaluate S_boundary(P) with linear interpolation and clamping."""
        return jnp.interp(P, self._P, self._S)


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
        """Look up a property with phase weighting (solid/melt blend).

        For S < S_sol: use solid table.
        For S > S_liq: use melt table.
        Between: linear blend by melt fraction.
        """
        phi = self.melt_fraction(P, S)
        solid_table, melt_table = self._get_tables(prop_name)

        # Each table clamps internally to its own domain
        val_solid = solid_table(P, S)
        val_melt = melt_table(P, S)

        # NaN-safe blend: avoid 0.0 * NaN = NaN
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
        """Density rho(P, S) [kg/m^3].

        Harmonic mean in the mushy zone (SPIDER convention).
        """
        phi = self.melt_fraction(P, S)
        rho_sol = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq = self._lookup_at_phase_boundary('density', P, 'melt')

        inv_rho = (
            phi / jnp.maximum(rho_liq, 1.0)
            + (1.0 - phi) / jnp.maximum(rho_sol, 1.0)
        )
        return 1.0 / jnp.maximum(inv_rho, 1e-30)

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
