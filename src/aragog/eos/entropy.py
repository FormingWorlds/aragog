"""Entropy-based EOS layer for PALEOS P-S tables.

Loads SPIDER-format P-S tables (pressure-entropy) and provides property
lookups (P, S) -> T, rho, Cp, alpha, dTdPs, phi, L(P). This is the EOS
backend for the entropy formulation of Aragog.

Table format (SPIDER convention):
    - Header line 1: # n_header n_pressure n_entropy
    - Header lines 2-4: comments
    - Header line 5: # P_scale S_scale quantity_scale
    - Data: 3 columns (P_nondim, S_nondim, quantity_nondim)
    - Separate files for solid and melt phases
    - Phase boundaries: solidus_P-S.dat, liquidus_P-S.dat (2 columns)

Files needed per material:
    temperature_{solid,melt}.dat
    density_{solid,melt}.dat
    heat_capacity_{solid,melt}.dat
    adiabat_temp_grad_{solid,melt}.dat  (dT/dP|_S, NOT nabla_ad)
    solidus_P-S.dat
    liquidus_P-S.dat
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def _bisect_left_clamp(arr: list[float], val: float) -> int:
    """Find left index for interpolation, clamped to valid range.

    Returns i such that arr[i] <= val <= arr[i+1], clamped to
    [0, len(arr)-2]. Uses bisect for O(log n) lookup.

    Parameters
    ----------
    arr : list of float
        Sorted grid values (converted from numpy to Python list).
    val : float
        Value to locate.

    Returns
    -------
    int
        Left index, in [0, len(arr)-2].
    """
    n = len(arr)
    idx = bisect.bisect_right(arr, val) - 1
    # Clamp to valid interpolation range
    if idx < 0:
        return 0
    if idx >= n - 1:
        return n - 2
    return idx


def _interp_bilinear_scalar(
    P_grid: list[float],
    S_grid: list[float],
    values: np.ndarray,
    P: float,
    S: float,
) -> float:
    """Pure-Python bilinear interpolation on a regular (P, S) grid.

    Avoids ALL numpy scalar operations, using only Python float arithmetic
    and direct ndarray element access via integer indexing. This is safe
    with numpy >= 2.4 where implicit array-to-scalar conversion raises
    TypeError.

    Parameters
    ----------
    P_grid : list of float
        Pressure grid values (sorted, ascending). Must be a Python list.
    S_grid : list of float
        Entropy grid values (sorted, ascending). Must be a Python list.
    values : ndarray, shape (n_P, n_S)
        Property values on the grid. Accessed via [int, int] indexing
        which returns a numpy scalar (0-d), then converted with .item().
    P : float
        Pressure query point (pure Python float).
    S : float
        Entropy query point (pure Python float).

    Returns
    -------
    float
        Interpolated value (pure Python float).
    """
    # Clamp to grid bounds
    P = max(P_grid[0], min(P, P_grid[-1]))
    S = max(S_grid[0], min(S, S_grid[-1]))

    ip = _bisect_left_clamp(P_grid, P)
    js = _bisect_left_clamp(S_grid, S)

    P_lo = P_grid[ip]
    P_hi = P_grid[ip + 1]
    S_lo = S_grid[js]
    S_hi = S_grid[js + 1]

    dP = P_hi - P_lo
    dS = S_hi - S_lo

    # Normalized distances (pure Python float arithmetic)
    tp = (P - P_lo) / dP if dP > 0.0 else 0.0
    ts = (S - S_lo) / dS if dS > 0.0 else 0.0

    # Four corner values. Use int() to ensure Python int indices,
    # and .item() to extract pure Python float from numpy scalar.
    v00 = float(values[int(ip), int(js)])
    v01 = float(values[int(ip), int(js + 1)])
    v10 = float(values[int(ip + 1), int(js)])
    v11 = float(values[int(ip + 1), int(js + 1)])

    # Bilinear blend (pure Python float arithmetic, no numpy)
    return (
        v00 * (1.0 - tp) * (1.0 - ts)
        + v01 * (1.0 - tp) * ts
        + v10 * tp * (1.0 - ts)
        + v11 * tp * ts
    )


def _interp_1d_scalar(
    x_grid: list[float], y_grid: list[float], x: float,
) -> float:
    """Pure-Python 1D linear interpolation with clamping.

    Parameters
    ----------
    x_grid : list of float
        Sorted x values (Python list).
    y_grid : list of float
        Corresponding y values (Python list).
    x : float
        Query point.

    Returns
    -------
    float
        Interpolated value (pure Python float).
    """
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    i = _bisect_left_clamp(x_grid, x)
    x0, x1 = x_grid[i], x_grid[i + 1]
    y0, y1 = y_grid[i], y_grid[i + 1]
    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    return y0 + t * (y1 - y0)


def _load_spider_ps_table(filepath: Path) -> dict:
    """Load a SPIDER-format P-S property table.

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    dict with keys:
        'P' : 1D array of pressure values [Pa]
        'S' : 1D array of entropy values [J/kg/K]
        'values' : 2D array of property values [SI], shape (n_P, n_S)
        'interp' : RegularGridInterpolator on (P, S)
    """
    with open(filepath) as f:
        header = f.readline().strip()
        parts = header.split()
        n_header = int(parts[1])
        n_P = int(parts[2])
        n_S = int(parts[3])
        # Read remaining header lines
        for _ in range(n_header - 1):
            line = f.readline()
        # Last header line has scaling factors
        scales_line = line.strip().lstrip('#').strip()
        scales = scales_line.split()
        P_scale = float(scales[0])
        S_scale = float(scales[1])
        Q_scale = float(scales[2])

    # Read data
    data = np.genfromtxt(filepath, skip_header=n_header)
    P_nondim = data[:, 0]
    S_nondim = data[:, 1]
    Q_nondim = data[:, 2]

    # Convert to SI
    P_all = P_nondim * P_scale
    S_all = S_nondim * S_scale
    Q_all = Q_nondim * Q_scale

    # Build unique grid vectors
    P_unique = np.unique(P_all)
    S_unique = np.unique(S_all)

    if len(P_unique) != n_P or len(S_unique) != n_S:
        logger.warning(
            'Table %s: expected %d x %d grid, got %d x %d unique values',
            filepath.name, n_P, n_S, len(P_unique), len(S_unique),
        )

    # Reshape to 2D grid. SPIDER writes S varying slowest (outer loop),
    # P varying fastest (inner loop). So the flat data is (n_S * n_P,)
    # with S blocks of n_P entries each. Reshape to (n_S, n_P) then
    # transpose to get (n_P, n_S) for RegularGridInterpolator((P, S)).
    values = Q_all.reshape(n_S, n_P).T

    interp = RegularGridInterpolator(
        (P_unique, S_unique), values,
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    return {
        'P': P_unique,
        'S': S_unique,
        'values': values,
        'interp': interp,
        'n_P': n_P,
        'n_S': n_S,
        # Python lists for pure-Python scalar interpolation (numpy 2.4 safe)
        'P_list': P_unique.tolist(),
        'S_list': S_unique.tolist(),
    }


def _load_spider_phase_boundary(filepath: Path) -> dict:
    """Load a SPIDER-format phase boundary file (solidus or liquidus in P-S).

    Returns
    -------
    dict with keys:
        'P' : 1D array [Pa]
        'S' : 1D array [J/kg/K]
        'interp' : callable P -> S
    """
    with open(filepath) as f:
        header = f.readline().strip()
        parts = header.split()
        n_header = int(parts[1])
        n_P = int(parts[2])
        for _ in range(n_header - 1):
            line = f.readline()
        scales_line = line.strip().lstrip('#').strip()
        scales = scales_line.split()
        P_scale = float(scales[0])
        S_scale = float(scales[1])

    data = np.genfromtxt(filepath, skip_header=n_header)
    P = data[:, 0] * P_scale
    S = data[:, 1] * S_scale

    from scipy.interpolate import interp1d
    interp = interp1d(P, S, kind='linear', bounds_error=False,
                      fill_value=(S[0], S[-1]))

    return {
        'P': P, 'S': S, 'interp': interp,
        'P_list': P.tolist(), 'S_list': S.tolist(),
    }


class EntropyEOS:
    """Entropy-based EOS from PALEOS P-S tables.

    Provides property lookups (P, S) -> T, rho, Cp, alpha, dTdPs, phi
    for a single mantle material with solid and melt phases.

    Parameters
    ----------
    eos_dir : Path or str
        Directory containing the SPIDER-format P-S table files.
    """

    def __init__(self, eos_dir: Path | str):
        eos_dir = Path(eos_dir)
        if not eos_dir.is_dir():
            raise FileNotFoundError(f'EOS directory not found: {eos_dir}')

        logger.info('Loading entropy EOS from %s', eos_dir)

        # Load property tables for solid and melt
        self._tables: dict[str, dict] = {}
        for phase in ('solid', 'melt'):
            self._tables[f'temperature_{phase}'] = _load_spider_ps_table(
                eos_dir / f'temperature_{phase}.dat')
            self._tables[f'density_{phase}'] = _load_spider_ps_table(
                eos_dir / f'density_{phase}.dat')
            self._tables[f'heat_capacity_{phase}'] = _load_spider_ps_table(
                eos_dir / f'heat_capacity_{phase}.dat')
            self._tables[f'dTdPs_{phase}'] = _load_spider_ps_table(
                eos_dir / f'adiabat_temp_grad_{phase}.dat')

            # Thermal expansivity: load from table if available (matches
            # SPIDER exactly), otherwise derived from thermodynamic identity.
            alpha_file = eos_dir / f'thermal_exp_{phase}.dat'
            if alpha_file.is_file():
                self._tables[f'thermal_exp_{phase}'] = _load_spider_ps_table(
                    alpha_file)
        self._has_alpha_tables = (
            'thermal_exp_solid' in self._tables
            and 'thermal_exp_melt' in self._tables
        )
        if self._has_alpha_tables:
            logger.info('Thermal expansivity loaded from P-S tables (SPIDER parity)')
        else:
            logger.info('Thermal expansivity will be derived from T, rho, Cp, dTdPs')

        # Load phase boundaries
        self._solidus = _load_spider_phase_boundary(eos_dir / 'solidus_P-S.dat')
        self._liquidus = _load_spider_phase_boundary(eos_dir / 'liquidus_P-S.dat')

        # Store P and S ranges (union of solid and melt tables)
        ref_melt = self._tables['temperature_melt']
        ref_solid = self._tables['temperature_solid']
        self.P_min = float(min(ref_melt['P'][0], ref_solid['P'][0]))
        self.P_max = float(max(ref_melt['P'][-1], ref_solid['P'][-1]))
        self.S_min = float(min(ref_melt['S'][0], ref_solid['S'][0]))
        self.S_max = float(max(ref_melt['S'][-1], ref_solid['S'][-1]))

        logger.info(
            'Entropy EOS loaded: P=[%.2e, %.2e] Pa, S=[%.0f, %.0f] J/kg/K, '
            '%d x %d grid (melt)',
            self.P_min, self.P_max, self.S_min, self.S_max,
            ref_melt['n_P'], ref_melt['n_S'],
        )

    def solidus_entropy(self, P: npt.NDArray | float) -> npt.NDArray:
        """Solidus entropy S_sol(P) [J/kg/K]."""
        return np.asarray(self._solidus['interp'](P), dtype=float)

    def liquidus_entropy(self, P: npt.NDArray | float) -> npt.NDArray:
        """Liquidus entropy S_liq(P) [J/kg/K]."""
        return np.asarray(self._liquidus['interp'](P), dtype=float)

    # ---- Pure-Python scalar methods (numpy 2.4 safe) ----
    # These avoid ALL numpy scalar conversions and are safe to call
    # inside scipy.optimize callbacks (brentq, etc.) where numpy 2.4
    # raises "only 0-dimensional arrays can be converted to Python
    # scalars" from C-extension float() calls on intermediate results.

    def _solidus_entropy_scalar(self, P: float) -> float:
        """Solidus entropy at a single pressure, pure Python."""
        return _interp_1d_scalar(
            self._solidus['P_list'], self._solidus['S_list'], P,
        )

    def _liquidus_entropy_scalar(self, P: float) -> float:
        """Liquidus entropy at a single pressure, pure Python."""
        return _interp_1d_scalar(
            self._liquidus['P_list'], self._liquidus['S_list'], P,
        )

    def _melt_fraction_scalar(self, P: float, S: float) -> float:
        """Melt fraction at a single (P, S) point, pure Python."""
        S_sol = self._solidus_entropy_scalar(P)
        S_liq = self._liquidus_entropy_scalar(P)
        dS = max(S_liq - S_sol, 1e-10)
        phi = (S - S_sol) / dS
        return max(0.0, min(1.0, phi))

    def _lookup_scalar(
        self, prop_name: str, phase: str, P: float, S: float,
    ) -> float:
        """Look up a single property value using pure-Python bilinear interp.

        Parameters
        ----------
        prop_name : str
            Property name (e.g. 'temperature', 'density').
        phase : str
            Phase name ('solid' or 'melt').
        P : float
            Pressure [Pa], pure Python float.
        S : float
            Entropy [J/kg/K], pure Python float.

        Returns
        -------
        float
            Property value (pure Python float).
        """
        table = self._tables[f'{prop_name}_{phase}']
        return _interp_bilinear_scalar(
            table['P_list'], table['S_list'], table['values'], P, S,
        )

    def temperature_scalar(self, P: float, S: float) -> float:
        """Temperature T(P, S) for a single point, pure Python.

        Uses the same phase-weighted blending as ``temperature()`` but
        with pure-Python arithmetic throughout. Safe for use inside
        scipy.optimize callbacks on numpy >= 2.4.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        S : float
            Entropy [J/kg/K].

        Returns
        -------
        float
            Temperature [K].
        """
        phi = self._melt_fraction_scalar(P, S)

        solid_table = self._tables['temperature_solid']
        melt_table = self._tables['temperature_melt']

        # Clamp to each table's own range
        S_sol_clamped = max(solid_table['S_list'][0],
                           min(S, solid_table['S_list'][-1]))
        S_melt_clamped = max(melt_table['S_list'][0],
                            min(S, melt_table['S_list'][-1]))
        P_sol_clamped = max(solid_table['P_list'][0],
                           min(P, solid_table['P_list'][-1]))
        P_melt_clamped = max(melt_table['P_list'][0],
                            min(P, melt_table['P_list'][-1]))

        val_solid = _interp_bilinear_scalar(
            solid_table['P_list'], solid_table['S_list'],
            solid_table['values'], P_sol_clamped, S_sol_clamped,
        )
        val_melt = _interp_bilinear_scalar(
            melt_table['P_list'], melt_table['S_list'],
            melt_table['values'], P_melt_clamped, S_melt_clamped,
        )

        # NaN-safe phase-weighted blend (same logic as _lookup_phase_weighted)
        result = 0.0
        if phi > 0.0:
            result += phi * val_melt
        if phi < 1.0:
            result += (1.0 - phi) * val_solid
        return result

    def melt_fraction(self, P: npt.NDArray | float,
                      S: npt.NDArray | float) -> npt.NDArray:
        """Melt fraction phi from position between solidus and liquidus.

        phi = 0 for S <= S_sol, phi = 1 for S >= S_liq, linear between.
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)
        dS = np.maximum(S_liq - S_sol, 1e-10)
        phi = np.clip((S - S_sol) / dS, 0.0, 1.0)
        return phi

    def _lookup_at_phase_boundary(
        self, prop_name: str, P: npt.NDArray, phase: str,
    ) -> npt.NDArray:
        """Look up a property at the solidus or liquidus for the given phase.

        This evaluates the solid table at S_solidus(P) or the melt table
        at S_liquidus(P), matching SPIDER's approach of using end-member
        properties at the phase boundaries (not at the actual S).
        """
        P = np.asarray(P, dtype=float)
        table = self._tables[f'{prop_name}_{phase}']
        # Clamp P to THIS table's range (not global range)
        P_clamped = np.clip(P, table['P'][0], table['P'][-1])

        if phase == 'solid':
            S_boundary = self.solidus_entropy(P)
        else:
            S_boundary = self.liquidus_entropy(P)

        S_clamped = np.clip(S_boundary, table['S'][0], table['S'][-1])
        pts = np.column_stack([P_clamped.ravel(), S_clamped.ravel()])
        return table['interp'](pts).reshape(P.shape)

    def _lookup_phase_weighted(
        self, prop_name: str, P: npt.NDArray, S: npt.NDArray,
    ) -> npt.NDArray:
        """Look up a property with phase weighting (solid/melt blend).

        For S < S_sol: use solid table.
        For S > S_liq: use melt table.
        Between: linear blend by melt fraction.
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        phi = self.melt_fraction(P, S)

        # Clamp S and P to each table's own range (not global range)
        # to avoid NaN from fill_value=nan outside individual table domains.
        solid_table = self._tables[f'{prop_name}_solid']
        melt_table = self._tables[f'{prop_name}_melt']

        S_solid_clamped = np.clip(S, solid_table['S'][0], solid_table['S'][-1])
        S_melt_clamped = np.clip(S, melt_table['S'][0], melt_table['S'][-1])
        P_solid_clamped = np.clip(P, solid_table['P'][0], solid_table['P'][-1])
        P_melt_clamped = np.clip(P, melt_table['P'][0], melt_table['P'][-1])

        pts_solid = np.column_stack([P_solid_clamped.ravel(), S_solid_clamped.ravel()])
        pts_melt = np.column_stack([P_melt_clamped.ravel(), S_melt_clamped.ravel()])

        val_solid = solid_table['interp'](pts_solid).reshape(P.shape)
        val_melt = melt_table['interp'](pts_melt).reshape(P.shape)

        # NaN-safe phase-weighted blend: avoid 0.0 * NaN = NaN
        result = np.where(phi > 0, phi * val_melt, 0.0) + \
                 np.where(phi < 1, (1.0 - phi) * val_solid, 0.0)
        return result

    def temperature(self, P: npt.NDArray | float,
                    S: npt.NDArray | float) -> npt.NDArray:
        """Temperature T(P, S) [K]."""
        return self._lookup_phase_weighted('temperature', P, S)

    def density(self, P: npt.NDArray | float,
                S: npt.NDArray | float) -> npt.NDArray:
        """Density rho(P, S) [kg/m^3].

        Uses harmonic mean in the mushy zone (SPIDER convention):
        1/rho = phi/rho_liq + (1-phi)/rho_sol.
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        phi = self.melt_fraction(P, S)

        rho_sol = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq = self._lookup_at_phase_boundary('density', P, 'melt')

        # Harmonic mean in mushy zone, pure phase outside
        inv_rho = phi / np.maximum(rho_liq, 1.0) + (1.0 - phi) / np.maximum(rho_sol, 1.0)
        return 1.0 / np.maximum(inv_rho, 1e-30)

    def heat_capacity(self, P: npt.NDArray | float,
                      S: npt.NDArray | float) -> npt.NDArray:
        """Specific heat capacity Cp(P, S) [J/kg/K]."""
        return self._lookup_phase_weighted('heat_capacity', P, S)

    def dTdPs(self, P: npt.NDArray | float,
              S: npt.NDArray | float) -> npt.NDArray:
        """Adiabatic temperature gradient dT/dP|_S (P, S) [K/Pa].

        This is the SPIDER convention: dT/dP along the adiabat, NOT
        the dimensionless nabla_ad = d ln T / d ln P.
        """
        return self._lookup_phase_weighted('dTdPs', P, S)

    def latent_heat(self, P: npt.NDArray | float) -> npt.NDArray:
        """Latent heat L(P) = T_fus × (S_liq - S_sol) [J/kg].

        P-dependent, following SPIDER convention. T_fus is the average
        of solidus and liquidus temperatures at the given pressure.
        """
        P = np.asarray(P, dtype=float)
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)
        T_sol = self._lookup_at_phase_boundary('temperature', P, 'solid')
        T_liq = self._lookup_at_phase_boundary('temperature', P, 'melt')
        T_fus = 0.5 * (T_sol + T_liq)
        return T_fus * np.maximum(S_liq - S_sol, 1.0)

    def thermal_expansivity(self, P: npt.NDArray | float,
                            S: npt.NDArray | float) -> npt.NDArray:
        """Thermal expansivity alpha(P, S) [1/K].

        When P-S thermal_exp tables are available (generated alongside
        SPIDER tables), uses them directly for exact parity with SPIDER.
        Otherwise, derives alpha from the thermodynamic identity:
        alpha = rho * Cp * |dT/dP|_S| / T.
        """
        if self._has_alpha_tables:
            return self._lookup_phase_weighted('thermal_exp', P, S)

        T = self.temperature(P, S)
        rho = self.density(P, S)
        Cp = self.heat_capacity(P, S)
        dTdPs_val = self.dTdPs(P, S)
        alpha = rho * Cp * np.abs(dTdPs_val) / np.maximum(T, 1.0)
        return alpha

    def invert_temperature(self, P: float, T_target: float) -> float:
        """Find entropy S such that T(P, S) = T_target.

        Uses Brent root-finding with pure-Python bilinear interpolation
        on the P-S temperature tables. All arithmetic inside the brentq
        callback uses only Python floats, avoiding numpy scalar
        conversions that break on numpy >= 2.4 (where ndim > 0 arrays
        cannot be implicitly converted to scalars).

        Parameters
        ----------
        P : float
            Pressure [Pa].
        T_target : float
            Target temperature [K].

        Returns
        -------
        float
            Entropy [J/kg/K] such that T(P, S) ~ T_target.

        Raises
        ------
        ValueError
            If T_target is outside the range of T(P, S) for this P.
        """
        from scipy.optimize import brentq

        # All values must be pure Python floats for numpy 2.4 safety.
        P_clamped = max(self.P_min, min(float(P), self.P_max))
        T_tgt = float(T_target)

        def residual(S_cand: float) -> float:
            # Pure-Python path: no numpy arrays created or converted.
            return self.temperature_scalar(P_clamped, S_cand) - T_tgt

        S_lo = self.S_min  # already Python float from __init__
        S_hi = self.S_max
        f_lo = residual(S_lo)
        f_hi = residual(S_hi)

        if f_lo * f_hi > 0:
            raise ValueError(
                f'Cannot invert T={T_target:.1f} K at P={P:.2e} Pa: '
                f'T(S_min={S_lo:.0f})={T_tgt+f_lo:.0f} K, '
                f'T(S_max={S_hi:.0f})={T_tgt+f_hi:.0f} K. '
                f'Target outside table range.'
            )

        S_root = brentq(residual, S_lo, S_hi, xtol=0.1, rtol=1e-10)
        # brentq returns a Python float (from C double), but be safe
        return float(S_root)
