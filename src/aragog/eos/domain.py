"""Valid domain of the P-S phase tables, read from the validity masks next to the tables.

A table generator can fill nodes that have no valid EOS value (outside the phase temperature
window) and mark them 0 in ``valid_mask_<phase>.dat``. A lookup on a filled node returns a
value with no EOS data behind it. The window of a pressure column is the entropy range from its
lowest to its highest valid node; between 2 columns the window ends are linear in pressure.
A 0 node between 2 valid nodes of a column is inside the window and is not judged. A column with
no valid node (a column the generator filled in full) has no window: a point at its pressure, or
between it and a neighbouring column, is infinitely outside.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aragog.utilities import blend_weight

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


class EOSDomainError(RuntimeError):
    """A lookup point of an accepted state is outside the valid window of its phase table."""


@dataclass(frozen=True)
class PhaseWindow:
    """Valid entropy window of one phase table per pressure column.

    Attributes
    ----------
    P : ndarray
        Pressure nodes of the table [Pa].
    lo, hi : ndarray
        Lowest and highest valid entropy of each column [J/kg/K]; +inf and -inf for a column
        with no valid node.
    dS : float
        Mean entropy spacing of the table [J/kg/K], the unit of ``distance``.
    """

    P: npt.NDArray
    lo: npt.NDArray
    hi: npt.NDArray
    dS: float

    @classmethod
    def from_mask(cls, P: npt.NDArray, S: npt.NDArray, mask: npt.NDArray) -> PhaseWindow:
        """Window ends from a mask of shape (n_S, n_P), 1 on a valid node and 0 on a filled one.

        Raises
        ------
        ValueError
            When an axis does not rise strictly, the mask shape is not (n_S, n_P), or a value is
            not 0 or 1.
        """
        P, S, mask = np.asarray(P, float), np.asarray(S, float), np.asarray(mask, float)
        if not (np.all(np.diff(P) > 0.0) and np.all(np.diff(S) > 0.0)):
            raise ValueError('the P and S axes of a mask must rise strictly')
        if mask.shape != (S.size, P.size):
            raise ValueError(f'mask shape {mask.shape} is not (n_S, n_P) = {(S.size, P.size)}')
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError('mask values must be 0 and 1')
        valid = mask.astype(bool)
        empty = ~valid.any(axis=0)
        i_lo, i_hi = valid.argmax(axis=0), S.size - 1 - valid[::-1].argmax(axis=0)
        n_inside_0 = int(np.where(empty, 0, i_hi - i_lo + 1 - valid.sum(axis=0)).sum())
        if n_inside_0:
            logger.info('%d mask nodes marked 0 lie inside the window, not judged', n_inside_0)
        if empty.any():
            logger.info('%d mask columns hold no valid node: no window there', int(empty.sum()))
        lo, hi = np.where(empty, np.inf, S[i_lo]), np.where(empty, -np.inf, S[i_hi])
        return cls(P=P, lo=lo, hi=hi, dS=float(np.mean(np.diff(S))))

    def ends(self, P: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Window ends at ``P``, linear between columns and held at the end columns; +inf and
        -inf where a column with no window takes part."""
        P = np.asarray(P, float)
        empty = ~np.isfinite(self.lo)
        lo = np.interp(P, self.P, np.where(empty, 0.0, self.lo))
        hi = np.interp(P, self.P, np.where(empty, 0.0, self.hi))
        if not empty.any():
            return lo, hi
        n = self.P.size
        k = np.clip(np.searchsorted(self.P, P, side='right') - 1, 0, n - 1)
        k1 = np.minimum(k + 1, n - 1)
        frac = np.clip(
            (P - self.P[k]) / np.where(k1 > k, self.P[k1] - self.P[k], 1.0), 0.0, 1.0
        )
        frac = np.where(k1 > k, frac, 0.0)  # the weight of column k1
        none = (empty[k] & (frac < 1.0)) | (empty[k1] & (frac > 0.0))
        return np.where(none, np.inf, lo), np.where(none, -np.inf, hi)

    def distance(self, P: npt.NDArray, S: npt.NDArray) -> npt.NDArray:
        """Distance of (P, S) outside the window in table S cells: 0 inside, inf with no window
        at P, NaN when not finite."""
        P, S = np.broadcast_arrays(np.asarray(P, float), np.asarray(S, float))
        lo, hi = self.ends(P)
        d = np.maximum(np.maximum(lo - S, S - hi), 0.0) / self.dS
        return np.where(np.isfinite(P) & np.isfinite(S), d, np.nan)


@dataclass(frozen=True)
class TableDomain:
    """Valid windows of the melt and solid tables."""

    melt: PhaseWindow
    solid: PhaseWindow

    @classmethod
    def from_dir(cls, eos_dir: Path | str, tables: dict) -> TableDomain | None:
        """Read ``valid_mask_melt.dat`` and ``valid_mask_solid.dat`` from ``eos_dir``.

        ``tables[phase]`` holds the axes ``'P'`` and ``'S'`` of the phase's temperature table.
        With neither mask file the check is off: None, with 1 info line.

        Raises
        ------
        FileNotFoundError
            When only one of the 2 mask files exists.
        ValueError
            When a mask breaks a rule of ``PhaseWindow.from_mask``.
        """
        files = {ph: Path(eos_dir) / f'valid_mask_{ph}.dat' for ph in ('melt', 'solid')}
        missing = [f for f in files.values() if not f.is_file()]
        if len(missing) == 2:
            logger.info('No valid_mask files in %s: the EOS domain check is off', eos_dir)
            return None
        if missing:
            raise FileNotFoundError(f'{missing[0]} is missing; the other phase has its mask')
        windows = {
            ph: PhaseWindow.from_mask(
                tables[ph]['P'], tables[ph]['S'], np.loadtxt(f, comments='#', ndmin=2)
            )
            for ph, f in files.items()
        }
        return cls(**windows)

    def lookup_distances(
        self, eos, P: npt.NDArray, S: npt.NDArray, smooth_width: float = 0.0
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Distance of the lookup points of (P, S) outside their table window, in S cells.

        A single phase is read at (P, S) on its own table; a mush point on the melt table at
        (P, S_liq(P)) and on the solid table at (P, S_sol(P)). A single-phase point where the
        phase blend weight (``aragog.utilities.blend_weight``) is above 0 also reads those 2
        phase-boundary lookups. The worst lookup of a point counts.

        Parameters
        ----------
        eos : EntropyEOS
            EOS with the phase boundaries and the melt fraction.
        P, S : ndarray
            Pressure [Pa] and entropy [J/kg/K] of the points.
        smooth_width : float, optional
            ``matprop_smooth_width`` of the phase evaluator [-]; 0 reads the phase boundaries
            in the mush only.

        Returns
        -------
        tuple of ndarray
            The distance (NaN for a non-finite point), True where the counted lookup is on the
            melt table, and the entropy of that lookup [J/kg/K].
        """
        P, S = np.broadcast_arrays(np.asarray(P, float), np.asarray(S, float))
        phi = np.asarray(eos._melt_fraction(P, S), float)
        s_liq = np.asarray(eos.liquidus_entropy(P), float)
        s_sol = np.asarray(eos.solidus_entropy(P), float)
        mush = (phi > 0.0) & (phi < 1.0)
        gphi = (S - s_sol) / np.maximum(s_liq - s_sol, 1e-10)
        band = mush | (blend_weight(gphi, smooth_width) > 0.0)
        d_own = np.where(phi >= 1.0, self.melt.distance(P, S), self.solid.distance(P, S))
        cand = np.stack(  # own table, liquidus on the melt table, solidus on the solid table
            [
                np.where(mush, -np.inf, d_own),
                np.where(band, self.melt.distance(P, s_liq), -np.inf),
                np.where(band, self.solid.distance(P, s_sol), -np.inf),
            ]
        )
        k = np.argmax(cand, axis=0)  # the first of equal distances: own table, then melt
        d = np.take_along_axis(cand, k[np.newaxis], axis=0)[0]
        d = np.where(np.isfinite(P) & np.isfinite(S), d, np.nan)
        melt_table = np.where(k == 0, phi >= 1.0, k == 1)
        return d, melt_table, np.choose(k, [S, s_liq, s_sol])

    @staticmethod
    def lookup_temperature(eos, P: float, S: float, melt: bool) -> float:
        """Temperature [K] that the melt (or solid) table returns for a lookup at (P, S).

        The point is clipped to the table axes as in the EOS lookups, with no range warning;
        NaN for a non-finite point.
        """
        if not (np.isfinite(P) and np.isfinite(S)):
            return math.nan
        return eos._lookup_scalar('temperature', 'melt' if melt else 'solid', P, S)
