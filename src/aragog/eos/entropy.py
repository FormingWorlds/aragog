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

logger = logging.getLogger('fwl.' + __name__)


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
    x_grid: list[float],
    y_grid: list[float],
    x: float,
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

    The PROTEUS-coupled path provides these tables via Zalmoxis (under
    ``<outdir>/data/aragog_pt/``) and PROTEUS standalone runs read them
    from ``$FWL_DATA``. They are not bundled with Aragog; an explicit
    file-existence check up front gives a clearer error than the bare
    ``open()`` raise from a missing path.

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

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist or is not a regular file.
    """
    fp = Path(filepath)
    if not fp.is_file():
        raise FileNotFoundError(
            f'EOS table not found: {fp}. Expected a SPIDER-format P-S table; '
            'PROTEUS-coupled runs get this from Zalmoxis under '
            '<outdir>/data/aragog_pt/, standalone runs from $FWL_DATA.'
        )
    with open(fp) as f:
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
            filepath.name,
            n_P,
            n_S,
            len(P_unique),
            len(S_unique),
        )

    # Reshape to 2D grid. SPIDER writes S varying slowest (outer loop),
    # P varying fastest (inner loop). So the flat data is (n_S * n_P,)
    # with S blocks of n_P entries each. Reshape to (n_S, n_P) then
    # transpose to get (n_P, n_S) for RegularGridInterpolator((P, S)).
    values = Q_all.reshape(n_S, n_P).T

    # Bilinear interpolation (C^0), matching SPIDER's lookup method.
    # Cubic (C^2) was tried but introduces sub-percent property offsets
    # that break the Jconv-Jmix cancellation in the mushy zone, causing
    # the CMB cell to drain. SPIDER uses bilinear for all P-S lookups.
    interp = RegularGridInterpolator(
        (P_unique, S_unique),
        values,
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
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
        for _ in range(n_header - 1):
            line = f.readline()
        scales_line = line.strip().lstrip('#').strip()
        scales = scales_line.split()
        P_scale = float(scales[0])
        S_scale = float(scales[1])

    data = np.genfromtxt(filepath, skip_header=n_header)
    P = data[:, 0] * P_scale
    S = data[:, 1] * S_scale

    # Linear interpolation matching SPIDER's 1D phase boundary lookup.
    # PchipInterpolator (C^1 monotone cubic Hermite) was tried but its
    # sub-percent offsets at phase boundaries compound into the Jconv-Jmix
    # cancellation failure that drains the CMB cell. SPIDER uses plain
    # linear interpolation for solidus/liquidus S(P).
    from scipy.interpolate import interp1d

    _lin = interp1d(
        P, S, kind='linear', bounds_error=False, fill_value=(float(S[0]), float(S[-1]))
    )

    # Finite-difference derivative for dS/dP (matches linear segments).
    _dSdP = np.gradient(S, P)
    _dlin = interp1d(P, _dSdP, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))

    def interp(Pq):
        Pq_arr = np.asarray(Pq, dtype=float)
        out = np.asarray(_lin(Pq_arr), dtype=float)
        return out.reshape(Pq_arr.shape)

    def dinterp(Pq):
        """dS/dP at the given pressure(s), in J/(kg·K·Pa).

        Outside the tabulated range, falls back to zero (the endpoint
        entropy is clamped by ``interp`` so its derivative is zero).
        """
        Pq_arr = np.asarray(Pq, dtype=float)
        out = np.asarray(_dlin(Pq_arr), dtype=float)
        return out.reshape(Pq_arr.shape)

    return {
        'P': P,
        'S': S,
        'interp': interp,
        'dinterp': dinterp,
        'P_list': P.tolist(),
        'S_list': S.tolist(),
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

        logger.debug('Loading entropy EOS from %s', eos_dir)

        # Load property tables for solid and melt
        self._tables: dict[str, dict] = {}
        for phase in ('solid', 'melt'):
            self._tables[f'temperature_{phase}'] = _load_spider_ps_table(
                eos_dir / f'temperature_{phase}.dat'
            )
            self._tables[f'density_{phase}'] = _load_spider_ps_table(
                eos_dir / f'density_{phase}.dat'
            )
            self._tables[f'heat_capacity_{phase}'] = _load_spider_ps_table(
                eos_dir / f'heat_capacity_{phase}.dat'
            )
            self._tables[f'dTdPs_{phase}'] = _load_spider_ps_table(
                eos_dir / f'adiabat_temp_grad_{phase}.dat'
            )

            # Thermal expansivity: load from table if available (matches
            # SPIDER exactly), otherwise derived from thermodynamic identity.
            alpha_file = eos_dir / f'thermal_exp_{phase}.dat'
            if alpha_file.is_file():
                self._tables[f'thermal_exp_{phase}'] = _load_spider_ps_table(alpha_file)
        self._has_alpha_tables = (
            'thermal_exp_solid' in self._tables and 'thermal_exp_melt' in self._tables
        )
        if self._has_alpha_tables:
            logger.debug('Thermal expansivity loaded from P-S tables (SPIDER parity)')
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

        logger.debug(
            'Entropy EOS loaded: P=[%.2e, %.2e] Pa, S=[%.0f, %.0f] J/kg/K, %d x %d grid (melt)',
            self.P_min,
            self.P_max,
            self.S_min,
            self.S_max,
            ref_melt['n_P'],
            ref_melt['n_S'],
        )

        self._build_enthalpy_table()

    def _build_enthalpy_table(self) -> None:
        """Precompute specific enthalpy h(P, S) on the table grid.

        Uses the entropy form of the fundamental thermodynamic relation,
        ``dh = T dS + (1/rho) dP``, integrated along a fixed path:
        first an isobar at the lowest tabulated pressure (S varying
        from S_grid[0] to the query S), then an isentrope at the query
        S (P varying from P_grid[0] to the query P). Anchor enthalpy
        at the table corner is set to zero. Because dh is exact, the
        integral is path-independent, but this particular path uses
        only table-coordinate values that the loader has already
        evaluated, which is what makes the precomputation cheap.

        The absolute level of the resulting table is anchor-dependent
        and therefore arbitrary; only differences h(P_a, S_a) - h(P_b,
        S_b) and the time derivatives that follow from them are
        physical. The conservation diagnostic that consumes this table
        cancels the anchor identically.

        Latent heat enters automatically: across the mushy zone at
        constant P the temperature on the integration isobar is held
        at the local melting temperature while S sweeps from solidus
        to liquidus, so the contribution to h equals T_phase * (S_liq
        - S_sol), which is exactly the latent heat per unit mass.
        """
        # Pick the integration grid: take pressure from either table
        # (they share the same P axis in the SPIDER convention) and
        # entropy from the union of solid and melt S axes so the table
        # spans the full physical range from cold solid (S near
        # solidus_S(P_min)) up to superheated melt. Using only the
        # melt table's S grid would silently clamp solid-phase queries
        # to the lowest melt-table S and report identical h for any
        # cell below that threshold.
        P_grid = np.asarray(self._tables['temperature_solid']['P'], dtype=float)
        S_grid = np.unique(
            np.concatenate(
                [
                    np.asarray(self._tables['temperature_solid']['S'], dtype=float),
                    np.asarray(self._tables['temperature_melt']['S'], dtype=float),
                ]
            )
        )
        n_P, n_S = len(P_grid), len(S_grid)

        # ── Step 1: isobar h(P0, S) = ∫_{S_grid[0]}^{S} T(P0, S') dS'
        P0 = float(P_grid[0])
        T_isobar = np.asarray(self.temperature(np.full(n_S, P0), S_grid)).ravel()
        h_isobar = np.zeros(n_S)
        h_isobar[1:] = np.cumsum(0.5 * (T_isobar[:-1] + T_isobar[1:]) * np.diff(S_grid))

        # ── Step 2: isentrope h(P, S_j) = h_isobar[j] + ∫_{P0}^P 1/rho dP
        h_grid = np.empty((n_P, n_S))
        for j in range(n_S):
            S_iso = np.full(n_P, S_grid[j])
            rho_iso = np.asarray(self.density(P_grid, S_iso)).ravel()
            inv_rho = 1.0 / np.maximum(rho_iso, 1.0)
            seg = 0.5 * (inv_rho[:-1] + inv_rho[1:]) * np.diff(P_grid)
            h_grid[0, j] = h_isobar[j]
            h_grid[1:, j] = h_isobar[j] + np.cumsum(seg)

        self._h_grid_P = P_grid
        self._h_grid_S = S_grid
        self._h_grid_P_list = P_grid.tolist()
        self._h_grid_S_list = S_grid.tolist()
        self._h_grid = h_grid
        self._h_interp = RegularGridInterpolator(
            (P_grid, S_grid),
            h_grid,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        logger.debug(
            'EOS-consistent enthalpy table built: %d x %d grid, '
            'anchor h(P=%.2e, S=%.0f) = 0 J/kg, '
            'h(P_max, S_max) = %.3e J/kg',
            n_P,
            n_S,
            P0,
            float(S_grid[0]),
            float(h_grid[-1, -1]),
        )

    def specific_enthalpy(
        self,
        P: npt.NDArray | float,
        S: npt.NDArray | float,
    ) -> npt.NDArray:
        """EOS-consistent specific enthalpy h(P, S) [J/kg].

        Bilinear interpolation on the precomputed table built in
        ``_build_enthalpy_table``. Inputs outside the table P or S
        range are clamped to the nearest table edge so the lookup
        never returns NaN; callers querying out-of-range states
        receive the boundary value instead of an exception.

        Parameters
        ----------
        P : array or float
            Pressure [Pa].
        S : array or float
            Entropy [J/kg/K].

        Returns
        -------
        ndarray
            Specific enthalpy [J/kg], same shape as broadcast(P, S).
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        P_clamped = np.clip(P, self._h_grid_P[0], self._h_grid_P[-1])
        S_clamped = np.clip(S, self._h_grid_S[0], self._h_grid_S[-1])
        pts = np.column_stack([P_clamped.ravel(), S_clamped.ravel()])
        return self._h_interp(pts).reshape(P_clamped.shape)

    def _specific_enthalpy_scalar(self, P: float, S: float) -> float:
        """Specific enthalpy at a single (P, S) point, pure Python.

        Used inside scalar callbacks where numpy >= 2.4 forbids implicit
        ndarray-to-scalar conversion. Same pattern as
        ``temperature_scalar``.
        """
        return _interp_bilinear_scalar(
            self._h_grid_P_list,
            self._h_grid_S_list,
            self._h_grid,
            P,
            S,
        )

    def solidus_entropy(self, P: npt.NDArray | float) -> npt.NDArray:
        """Solidus entropy S_sol(P) [J/kg/K]."""
        return np.asarray(self._solidus['interp'](P), dtype=float)

    def liquidus_entropy(self, P: npt.NDArray | float) -> npt.NDArray:
        """Liquidus entropy S_liq(P) [J/kg/K]."""
        return np.asarray(self._liquidus['interp'](P), dtype=float)

    def solidus_entropy_dP(self, P: npt.NDArray | float) -> npt.NDArray:
        """dS_sol/dP at the given pressure(s), in J/(kg·K·Pa).

        Needed by the SPIDER-parity mixing-flux formula in
        ``entropy_state.update``. SPIDER computes Jmix via a bracket
        expression ``dS/dr − [φ dS_liq/dP + (1−φ) dS_sol/dP] dP/dr``
        (``energy.c::GetMixingHeatFlux`` lines 307-309). Exposing the
        phase-boundary P-derivatives here lets Aragog match SPIDER's
        formula exactly without resorting to un-truncated gphi
        arithmetic.
        """
        return np.asarray(self._solidus['dinterp'](P), dtype=float)

    def liquidus_entropy_dP(self, P: npt.NDArray | float) -> npt.NDArray:
        """dS_liq/dP at the given pressure(s), in J/(kg·K·Pa)."""
        return np.asarray(self._liquidus['dinterp'](P), dtype=float)

    # ---- Pure-Python scalar methods (numpy 2.4 safe) ----
    # These avoid ALL numpy scalar conversions and are safe to call
    # inside scipy.optimize callbacks (brentq, etc.) where numpy 2.4
    # raises "only 0-dimensional arrays can be converted to Python
    # scalars" from C-extension float() calls on intermediate results.

    def _solidus_entropy_scalar(self, P: float) -> float:
        """Solidus entropy at a single pressure, pure Python."""
        return _interp_1d_scalar(
            self._solidus['P_list'],
            self._solidus['S_list'],
            P,
        )

    def _liquidus_entropy_scalar(self, P: float) -> float:
        """Liquidus entropy at a single pressure, pure Python."""
        return _interp_1d_scalar(
            self._liquidus['P_list'],
            self._liquidus['S_list'],
            P,
        )

    def _melt_fraction_scalar(self, P: float, S: float) -> float:
        """Melt fraction at a single (P, S) point, pure Python."""
        S_sol = self._solidus_entropy_scalar(P)
        S_liq = self._liquidus_entropy_scalar(P)
        dS = max(S_liq - S_sol, 1e-10)
        phi = (S - S_sol) / dS
        return max(0.0, min(1.0, phi))

    def _lookup_scalar(
        self,
        prop_name: str,
        phase: str,
        P: float,
        S: float,
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
            table['P_list'],
            table['S_list'],
            table['values'],
            P,
            S,
        )

    def temperature_scalar(self, P: float, S: float) -> float:
        """Temperature T(P, S) for a single point, pure Python.

        Uses the same Lever Rule blending as ``temperature()`` but
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

        # Lever Rule: in the mushy zone, evaluate at phase boundary
        # entropies; outside, evaluate at actual S.
        if 0.0 < phi < 1.0:
            S_for_solid = self._solidus_entropy_scalar(P)
            S_for_melt = self._liquidus_entropy_scalar(P)
        else:
            S_for_solid = S
            S_for_melt = S

        # Clamp to each table's own range
        S_sol_clamped = max(
            solid_table['S_list'][0], min(S_for_solid, solid_table['S_list'][-1])
        )
        S_melt_clamped = max(melt_table['S_list'][0], min(S_for_melt, melt_table['S_list'][-1]))
        P_sol_clamped = max(solid_table['P_list'][0], min(P, solid_table['P_list'][-1]))
        P_melt_clamped = max(melt_table['P_list'][0], min(P, melt_table['P_list'][-1]))

        val_solid = _interp_bilinear_scalar(
            solid_table['P_list'],
            solid_table['S_list'],
            solid_table['values'],
            P_sol_clamped,
            S_sol_clamped,
        )
        val_melt = _interp_bilinear_scalar(
            melt_table['P_list'],
            melt_table['S_list'],
            melt_table['values'],
            P_melt_clamped,
            S_melt_clamped,
        )

        # NaN-safe phase-weighted blend
        result = 0.0
        if phi > 0.0:
            result += phi * val_melt
        if phi < 1.0:
            result += (1.0 - phi) * val_solid
        return result

    def melt_fraction(self, P: npt.NDArray | float, S: npt.NDArray | float) -> npt.NDArray:
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
        self,
        prop_name: str,
        P: npt.NDArray,
        phase: str,
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
        self,
        prop_name: str,
        P: npt.NDArray,
        S: npt.NDArray,
    ) -> npt.NDArray:
        """Look up a property with Lever Rule blending (SPIDER parity).

        In the two-phase zone (S_sol < S < S_liq), SPIDER evaluates
        end-member properties at the phase boundary entropies, not at
        the actual cell entropy (eos_composite.c:216-224):

            val = phi * val_melt(P, S_liquidus)
                + (1 - phi) * val_solid(P, S_solidus)

        This is the Lever Rule: properties in the mushy zone are
        interpolated between the two coexistence values at that
        pressure, weighted by melt fraction.

        Outside the two-phase zone the single-phase table is evaluated
        at the actual (P, S).
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        phi = self.melt_fraction(P, S)

        solid_table = self._tables[f'{prop_name}_solid']
        melt_table = self._tables[f'{prop_name}_melt']

        # Phase boundary entropies for the Lever Rule
        S_sol = self.solidus_entropy(P)
        S_liq = self.liquidus_entropy(P)

        # In the mushy zone (0 < phi < 1): evaluate at the phase
        # boundary entropies (Lever Rule). Outside: evaluate at
        # actual S (single-phase regime).
        S_for_solid = np.where((phi > 0) & (phi < 1), S_sol, S)
        S_for_melt = np.where((phi > 0) & (phi < 1), S_liq, S)

        # Clamp to each table's own range
        S_solid_clamped = np.clip(S_for_solid, solid_table['S'][0], solid_table['S'][-1])
        S_melt_clamped = np.clip(S_for_melt, melt_table['S'][0], melt_table['S'][-1])
        P_solid_clamped = np.clip(P, solid_table['P'][0], solid_table['P'][-1])
        P_melt_clamped = np.clip(P, melt_table['P'][0], melt_table['P'][-1])

        pts_solid = np.column_stack([P_solid_clamped.ravel(), S_solid_clamped.ravel()])
        pts_melt = np.column_stack([P_melt_clamped.ravel(), S_melt_clamped.ravel()])

        val_solid = solid_table['interp'](pts_solid).reshape(P.shape)
        val_melt = melt_table['interp'](pts_melt).reshape(P.shape)

        # NaN-safe phase-weighted blend: avoid 0.0 * NaN = NaN
        result = np.where(phi > 0, phi * val_melt, 0.0) + np.where(
            phi < 1, (1.0 - phi) * val_solid, 0.0
        )
        return result

    def temperature(self, P: npt.NDArray | float, S: npt.NDArray | float) -> npt.NDArray:
        """Temperature T(P, S) [K]."""
        return self._lookup_phase_weighted('temperature', P, S)

    def density(self, P: npt.NDArray | float, S: npt.NDArray | float) -> npt.NDArray:
        """Density rho(P, S) [kg/m^3].

        In the mushy zone (0 < phi < 1): harmonic mean of end-member
        densities evaluated at phase boundary entropies (SPIDER
        eos_composite.c:236-237).

        Outside the mushy zone (phi=0 or phi=1): single-phase table
        evaluated at the actual S (clamped to table range), matching
        SPIDER's combine_matprop(smth=0, mixed, single) path.
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)
        phi = self.melt_fraction(P, S)
        mushy = (phi > 0) & (phi < 1)

        solid_table = self._tables['density_solid']
        melt_table = self._tables['density_melt']

        # Mushy zone: evaluate at phase boundaries (Lever Rule)
        rho_sol_boundary = self._lookup_at_phase_boundary('density', P, 'solid')
        rho_liq_boundary = self._lookup_at_phase_boundary('density', P, 'melt')
        inv_rho_mushy = phi / np.maximum(rho_liq_boundary, 1.0) + (1.0 - phi) / np.maximum(
            rho_sol_boundary, 1.0
        )
        rho_mushy = 1.0 / np.maximum(inv_rho_mushy, 1e-30)

        # Single-phase: evaluate at actual S (clamped to table range)
        S_solid_clamped = np.clip(S, solid_table['S'][0], solid_table['S'][-1])
        S_melt_clamped = np.clip(S, melt_table['S'][0], melt_table['S'][-1])
        P_solid_clamped = np.clip(P, solid_table['P'][0], solid_table['P'][-1])
        P_melt_clamped = np.clip(P, melt_table['P'][0], melt_table['P'][-1])
        pts_solid = np.column_stack([P_solid_clamped.ravel(), S_solid_clamped.ravel()])
        pts_melt = np.column_stack([P_melt_clamped.ravel(), S_melt_clamped.ravel()])
        rho_solid_single = solid_table['interp'](pts_solid).reshape(P.shape)
        rho_melt_single = melt_table['interp'](pts_melt).reshape(P.shape)
        # 0.5 here is a binary table-selector for the non-mushy fallback
        # (phi is essentially 0 or 1 outside the mushy band); it is NOT
        # the rheological critical melt fraction. The RCMF is
        # ``EntropyPhaseEvaluator._phi_rheo`` and drives the viscosity
        # tanh blend separately.
        rho_single = np.where(phi >= 0.5, rho_melt_single, rho_solid_single)

        return np.where(mushy, rho_mushy, rho_single)

    def heat_capacity(self, P: npt.NDArray | float, S: npt.NDArray | float) -> npt.NDArray:
        """Specific heat capacity Cp(P, S) [J/kg/K].

        Linear blend of solid and melt P-S tables by melt fraction
        (the ``cp_blend = 'linear'`` convention). Use
        ``heat_capacity_latent_blend`` for SPIDER-parity mushy-zone
        Cp that includes the latent-heat contribution.
        """
        return self._lookup_phase_weighted('heat_capacity', P, S)

    def heat_capacity_latent_blend(
        self, P: npt.NDArray | float, S: npt.NDArray | float, width: float = 0.01
    ) -> npt.NDArray:
        """Latent-heat-augmented Cp(P, S), matching SPIDER convention.

        Mirrors SPIDER's ``eos_composite.c:226-232`` formula:

            Cp_mix = (S_liq - S_sol) / (T_liq - T_sol)
                   * (T_sol + 0.5 * (T_liq - T_sol))

        which captures the latent-heat contribution to apparent heat
        capacity in the mushy zone. Outside the phase boundary the
        result reduces to the pure-phase Cp via tanh smoothing.

        The plain linear blend used by ``heat_capacity`` above misses
        the latent-heat term and underestimates Cp by up to a factor
        of 6 in the deep mushy regime relative to SPIDER's internal
        cp_s. This formula closes that gap.

        Parameters
        ----------
        P : array or float
            Pressure [Pa].
        S : array or float
            Entropy [J/kg/K].
        width : float
            Tanh smoothing width across the phase boundary, in units
            of melt fraction. SPIDER default is 0.01.

        Returns
        -------
        ndarray
            Heat capacity [J/kg/K] with the same shape as the inputs.
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)

        # Phase boundaries at this pressure
        S_sol = np.asarray(self.solidus_entropy(P)).reshape(P.shape)
        S_liq = np.asarray(self.liquidus_entropy(P)).reshape(P.shape)
        T_sol = np.asarray(self._lookup_at_phase_boundary('temperature', P, 'solid')).reshape(
            P.shape
        )
        T_liq = np.asarray(self._lookup_at_phase_boundary('temperature', P, 'melt')).reshape(
            P.shape
        )

        dS = S_liq - S_sol
        dT = T_liq - T_sol

        # Mixed-phase Cp from the latent-heat formula
        # SPIDER: Cp_mix = (S_liq - S_sol) / (T_liq - T_sol) * T_mid
        safe_dT = np.where(np.abs(dT) > 1e-10, dT, 1e-10)
        T_mid = T_sol + 0.5 * dT
        Cp_mix = (dS / safe_dT) * T_mid

        # Pure-phase Cp from the linear-blend tables
        Cp_pure = self._lookup_phase_weighted('heat_capacity', P, S)

        # Un-truncated phase fraction (SPIDER's gphi convention)
        safe_dS = np.where(np.abs(dS) > 1e-10, dS, 1e-10)
        gphi = (S - S_sol) / safe_dS

        # SPIDER-parity smoothing (util.c:245-270, get_smoothing).
        # Bell shape centred on the mushy zone: rises at gphi=0 with
        # tanh(gphi/w), falls at gphi=1 with tanh((1-gphi)/w). The
        # width parameter w controls the shoulder into the single-
        # phase regime.
        #
        # IMPORTANT: SPIDER uses tanh(x/w), not expit(x/w) which
        # equals 0.5*(1+tanh(x/(2w))). The previous expit formula
        # had an effective width of w/2, making the transition twice
        # as narrow and producing 16% less latent-heat Cp at
        # crystallization onset (gphi=0.994).
        w = max(width, 1e-6)
        smooth = np.where(
            gphi > 0.5,
            0.5 * (1.0 + np.tanh((1.0 - gphi) / w)),  # fall at liquidus
            0.5 * (1.0 + np.tanh(gphi / w)),  # rise at solidus
        )

        return smooth * Cp_mix + (1.0 - smooth) * Cp_pure

    def thermal_expansivity_composite_blend(
        self,
        P: npt.NDArray | float,
        S: npt.NDArray | float,
        width: float = 0.01,
    ) -> npt.NDArray:
        """Composite two-phase thermal expansivity, matching SPIDER.

        Mirrors SPIDER's ``eos_composite.c:246`` formula:

            alpha_mix = (rho_solid - rho_melt) / (T_liq - T_sol) / rho

        where ``rho`` is the harmonic-mean composite density. This
        effective alpha captures the density change from the phase
        transition (Clausius-Clapeyron contribution), which dominates
        over the single-phase alpha by a factor of ~5 in the deep
        mushy zone. Outside the mushy band the result reduces to the
        pure-phase alpha from the P-S tables via tanh smoothing
        (SPIDER ``eos_composite.c:279``).

        Parameters
        ----------
        P : array or float
            Pressure [Pa].
        S : array or float
            Entropy [J/kg/K].
        width : float
            Tanh smoothing width across the phase boundary.

        Returns
        -------
        ndarray
            Thermal expansivity [1/K].
        """
        P = np.asarray(P, dtype=float)
        S = np.asarray(S, dtype=float)

        # Phase-boundary quantities at this pressure
        S_sol = np.asarray(self.solidus_entropy(P)).reshape(P.shape)
        S_liq = np.asarray(self.liquidus_entropy(P)).reshape(P.shape)
        T_sol = np.asarray(self._lookup_at_phase_boundary('temperature', P, 'solid')).reshape(
            P.shape
        )
        T_liq = np.asarray(self._lookup_at_phase_boundary('temperature', P, 'melt')).reshape(
            P.shape
        )
        rho_sol = np.asarray(self._lookup_at_phase_boundary('density', P, 'solid')).reshape(
            P.shape
        )
        rho_liq = np.asarray(self._lookup_at_phase_boundary('density', P, 'melt')).reshape(
            P.shape
        )

        # Composite density (harmonic mean, SPIDER eos_composite.c:236)
        phi = self.melt_fraction(P, S)
        inv_rho = phi / np.maximum(rho_liq, 1.0) + (1.0 - phi) / np.maximum(rho_sol, 1.0)
        rho_comp = 1.0 / np.maximum(inv_rho, 1e-30)

        # Mixed-phase alpha (SPIDER eos_composite.c:246)
        dT = T_liq - T_sol
        safe_dT = np.where(np.abs(dT) > 1e-10, dT, 1e-10)
        alpha_mix = (rho_sol - rho_liq) / safe_dT / rho_comp

        # Pure-phase alpha from the linear-blend tables
        alpha_pure = self._lookup_phase_weighted('thermal_exp', P, S)

        # SPIDER-parity tanh smoothing (same as heat_capacity_latent_blend)
        dS = S_liq - S_sol
        safe_dS = np.where(np.abs(dS) > 1e-10, dS, 1e-10)
        gphi = (S - S_sol) / safe_dS

        w = max(width, 1e-6)
        smooth = np.where(
            gphi > 0.5,
            0.5 * (1.0 + np.tanh((1.0 - gphi) / w)),
            0.5 * (1.0 + np.tanh(gphi / w)),
        )

        return smooth * alpha_mix + (1.0 - smooth) * alpha_pure

    def dTdPs(self, P: npt.NDArray | float, S: npt.NDArray | float) -> npt.NDArray:
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

    def thermal_expansivity(
        self, P: npt.NDArray | float, S: npt.NDArray | float
    ) -> npt.NDArray:
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
                f'T(S_min={S_lo:.0f})={T_tgt + f_lo:.0f} K, '
                f'T(S_max={S_hi:.0f})={T_tgt + f_hi:.0f} K. '
                f'Target outside table range.'
            )

        S_root = brentq(residual, S_lo, S_hi, xtol=0.1, rtol=1e-10)
        # brentq returns a Python float (from C double), but be safe
        return float(S_root)
