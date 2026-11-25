"""Valid domain of the P-S phase tables, read from the validity masks next to the tables.

A table generator can fill nodes that have no valid EOS value (outside the phase temperature
window) and mark them 0 in ``valid_mask_<phase>.dat``. A lookup on a filled node returns a
value with no EOS data behind it. The window of a pressure column is the entropy range of its
valid nodes; between 2 columns the window ends are linear in pressure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

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
        Lowest and highest valid entropy of each column [J/kg/K].
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
            When the mask shape is not (n_S, n_P), a value is not 0 or 1, a column holds no
            valid node, or the valid nodes of a column are not one run.
        """
        P, S, mask = np.asarray(P, float), np.asarray(S, float), np.asarray(mask, float)
        if mask.shape != (S.size, P.size):
            raise ValueError(f'mask shape {mask.shape} is not (n_S, n_P) = {(S.size, P.size)}')
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError('mask values must be 0 and 1')
        valid = mask.astype(bool)
        lo, hi = np.empty(P.size), np.empty(P.size)
        for k in range(P.size):
            idx = np.flatnonzero(valid[:, k])
            if idx.size == 0:
                raise ValueError(f'mask column {k} holds no valid node')
            if idx[-1] - idx[0] + 1 != idx.size:
                raise ValueError(f'mask column {k} has a gap in its valid nodes')
            lo[k], hi[k] = S[idx[0]], S[idx[-1]]
        return cls(P=P, lo=lo, hi=hi, dS=float(np.mean(np.diff(S))))

    def ends(self, P: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Window ends at ``P``, linear between columns and held at the end columns."""
        P = np.asarray(P, float)
        return np.interp(P, self.P, self.lo), np.interp(P, self.P, self.hi)

    def distance(self, P: npt.NDArray, S: npt.NDArray) -> npt.NDArray:
        """Distance of (P, S) outside the window in table S cells: 0 inside, NaN when not finite."""
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
        Without both mask files the check is off: None, with 1 info line.

        Raises
        ------
        FileNotFoundError
            When only one of the 2 mask files exists.
        """
        files = {ph: Path(eos_dir) / f'valid_mask_{ph}.dat' for ph in ('melt', 'solid')}
        present = [f.is_file() for f in files.values()]
        if not any(present):
            logger.info('No valid_mask files in %s: the EOS domain check is off', eos_dir)
            return None
        if not all(present):
            missing = next(f for f, p in zip(files.values(), present) if not p)
            raise FileNotFoundError(f'{missing} is missing; the other phase has its mask')
        windows = {
            ph: PhaseWindow.from_mask(
                tables[ph]['P'], tables[ph]['S'], np.loadtxt(f, comments='#', ndmin=2)
            )
            for ph, f in files.items()
        }
        return cls(**windows)

    def lookup_distances(
        self, eos, P: npt.NDArray, S: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Distance of the lookup points of (P, S) outside their table window, in S cells.

        A single phase is read at (P, S) on its own table; a mush point on the melt table at
        (P, S_liq(P)) and on the solid table at (P, S_sol(P)), and the worse of the 2 counts.

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
        s_melt, s_solid = np.where(mush, s_liq, S), np.where(mush, s_sol, S)
        d_melt, d_solid = self.melt.distance(P, s_melt), self.solid.distance(P, s_solid)
        melt_table = np.where(mush, d_melt >= d_solid, phi >= 1.0)
        d = np.where(melt_table, d_melt, d_solid)
        d = np.where(np.isfinite(P) & np.isfinite(S), d, np.nan)
        return d, melt_table, np.where(melt_table, s_melt, s_solid)
