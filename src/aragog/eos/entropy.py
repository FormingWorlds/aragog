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

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


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

    return {'P': P, 'S': S, 'interp': interp}


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

        Uses Brent root-finding on the P-S temperature table. The
        temperature is monotonically increasing with entropy at fixed P
        (higher entropy = hotter), so the root is unique.

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

        P_clamped = float(np.clip(P, self.P_min, self.P_max))

        def residual(S_cand):
            T_eval = float(self.temperature(
                np.array([P_clamped]), np.array([S_cand]),
            ))
            return T_eval - T_target

        # Bracket: search from S_min to S_max
        S_lo, S_hi = self.S_min, self.S_max
        f_lo = residual(S_lo)
        f_hi = residual(S_hi)

        if f_lo * f_hi > 0:
            raise ValueError(
                f'Cannot invert T={T_target:.1f} K at P={P:.2e} Pa: '
                f'T(S_min={S_lo:.0f})={T_target+f_lo:.0f} K, '
                f'T(S_max={S_hi:.0f})={T_target+f_hi:.0f} K. '
                f'Target outside table range.'
            )

        S_root = brentq(residual, S_lo, S_hi, xtol=0.1, rtol=1e-10)
        return float(S_root)
