"""Liquid iron equation-of-state table of the core.

The table is the liquid side of the PALEOS iron EOS (the Luo et al. 2024 liquid, extended
metastably below the melting line) that Zalmoxis writes as
``2phase_iron_liquid_<sha>_<build>_<grid>.dat``: a full rectangle in pressure and temperature,
log-uniform in both, with the columns ``P[Pa] T[K] rho[kg/m^3] u[J/kg] s[J/(kg K)]
cp[J/(kg K)] cv[J/(kg K)] alpha[1/K] nabla_ad[-] phase`` and a header whose ``# Kind:`` line
names the branch. The same arrays can come as an ``.npz`` with the axes ``P_Pa`` and ``T_K``,
one 2-D array ``[P index, T index]`` per field and the header lines in ``header``.

Each field is interpolated by a bicubic spline in ``(log10 P, log10 T)``, exact at the nodes and
with continuous first derivatives, so that the isentrope and the heat capacity read from it
have no kinks at the grid lines. Everything here is NumPy and runs once, when a core profile is
installed; the solver reads only the tabulated results.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
from scipy.interpolate import RectBivariateSpline

FIELDS = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
_POSITIVE = ('rho', 'cp', 'cv')
_KIND = re.compile(r'^#\s*Kind:\s*(\w+)', re.MULTILINE)


class LiquidIronTable:
    """Bicubic interpolant of the PALEOS liquid iron table.

    Parameters
    ----------
    pressure : array_like
        Pressure axis [Pa], positive and strictly increasing, at least 4 points.
    temperature : array_like
        Temperature axis [K], positive and strictly increasing, at least 4 points.
    fields : dict
        One array of shape ``(len(pressure), len(temperature))`` per name in ``FIELDS``, in
        the units of the module docstring.
    header : str
        Header lines of the source table; its ``# Kind:`` line must name ``liquid``.

    Raises
    ------
    ValueError
        On a header without a liquid ``Kind`` line, a bad axis, a missing field, a field of
        the wrong shape, a non-finite cell, or a non-positive ``rho``, ``cp`` or ``cv``.
    """

    def __init__(self, pressure, temperature, fields: dict, header: str) -> None:
        kind = _KIND.search(header)
        if kind is None:
            raise ValueError('liquid iron table header has no Kind line')
        if kind.group(1) != 'liquid':
            raise ValueError(
                f"liquid iron table has kind {kind.group(1)!r}; the core needs kind 'liquid'"
            )
        axes = {
            'pressure': np.asarray(pressure, float),
            'temperature': np.asarray(temperature, float),
        }
        for name, ax in axes.items():
            if ax.ndim != 1 or ax.size < 4:
                raise ValueError(f'{name} axis needs at least 4 points, got shape {ax.shape}')
            if not np.all(ax > 0.0):
                raise ValueError(f'{name} axis must be positive')
            if not np.all(np.diff(ax) > 0.0):
                raise ValueError(f'{name} axis must be strictly increasing')
        missing = sorted(set(FIELDS) - set(fields))
        if missing:
            raise ValueError(f'liquid iron table is missing fields {missing}')
        shape = (axes['pressure'].size, axes['temperature'].size)
        self.fields = {}
        for f in FIELDS:
            arr = np.asarray(fields[f], float)
            if arr.shape != shape:
                raise ValueError(f'{f} has shape {arr.shape}, expected {shape}')
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'non-finite cell in {f}')
            if f in _POSITIVE and not np.all(arr > 0.0):
                raise ValueError(f'{f} must be positive in every cell')
            self.fields[f] = arr
        self.pressure, self.temperature = axes['pressure'], axes['temperature']
        self.header = header
        lp, lt = np.log10(self.pressure), np.log10(self.temperature)
        self._splines = {
            f: RectBivariateSpline(lp, lt, v, kx=3, ky=3) for f, v in self.fields.items()
        }

    @classmethod
    def from_file(cls, path) -> LiquidIronTable:
        """Read a Zalmoxis ``.dat`` table or its ``.npz`` cut.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            On another suffix, an ``.npz`` without one of its keys, a text table without data
            rows, with a row of fewer than 9 numbers or that is not a full rectangle in P and
            T, or any refusal of the constructor.
        """
        path = pathlib.Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'liquid iron table not found: {path}')
        if path.suffix == '.npz':
            with np.load(path) as d:
                missing = sorted({'P_Pa', 'T_K', 'header', *FIELDS} - set(d.files))
                if missing:
                    raise ValueError(f'{path.name} is missing {missing}')
                return cls(d['P_Pa'], d['T_K'], {f: d[f] for f in FIELDS}, str(d['header']))
        if path.suffix != '.dat':
            raise ValueError(f'liquid iron table must be .dat or .npz, got {path.name}')
        lines = path.read_text(encoding='utf-8').splitlines()
        header = '\n'.join(ln for ln in lines if ln.startswith('#'))
        rows = [ln.split()[:9] for ln in lines if ln.strip() and not ln.startswith('#')]
        if not rows:
            raise ValueError(f'{path.name} has no data rows')
        try:
            rows = np.array(rows, dtype=float).reshape(len(rows), 9)
        except ValueError:
            raise ValueError(
                f'{path.name}: every row needs 9 numbers (P, T, 7 fields)'
            ) from None
        p_ax, t_ax = np.unique(rows[:, 0]), np.unique(rows[:, 1])
        grid = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        full = p_ax.size * t_ax.size == rows.shape[0]
        grid = grid.reshape(p_ax.size, t_ax.size, 9) if full else grid
        if not full or not (
            np.all(grid[:, :, 0] == p_ax[:, None]) and np.all(grid[:, :, 1] == t_ax[None, :])
        ):
            raise ValueError(f'{path.name} is not a full rectangle in P and T')
        return cls(p_ax, t_ax, {f: grid[:, :, k + 2] for k, f in enumerate(FIELDS)}, header)

    def contains(self, pressure, temperature):
        """True where ``(pressure, temperature)`` [Pa, K] lies inside the table box."""
        p, t = np.asarray(pressure, float), np.asarray(temperature, float)
        return (
            (p >= self.pressure[0])
            & (p <= self.pressure[-1])
            & (t >= self.temperature[0])
            & (t <= self.temperature[-1])
        )

    def _logs(self, pressure, temperature):
        p, t = np.broadcast_arrays(np.asarray(pressure, float), np.asarray(temperature, float))
        inside = self.contains(p, t)
        if not np.all(inside):
            k = np.flatnonzero(~inside.ravel())[0]
            raise ValueError(
                f'P = {p.ravel()[k]:.4e} Pa, T = {t.ravel()[k]:.1f} K lies outside the liquid '
                f'iron table ({self.pressure[0]:.3e} to {self.pressure[-1]:.3e} Pa, '
                f'{self.temperature[0]:.1f} to {self.temperature[-1]:.1f} K)'
            )
        return np.log10(p), np.log10(t), p.shape

    def value(self, field: str, pressure, temperature):
        """Interpolated ``field`` at ``(pressure, temperature)`` [Pa, K], in its table unit.

        Raises
        ------
        ValueError
            For a point outside the table box.
        """
        return self._ev(field, pressure, temperature, 0, 0)

    def ds_dT(self, pressure, temperature):
        """``(ds/dT)_P`` [J kg-1 K-2] of the interpolant at ``(pressure, temperature)``."""
        return self._ev('s', pressure, temperature, 0, 1)

    def ds_dP(self, pressure, temperature):
        """``(ds/dP)_T`` [J kg-1 K-1 Pa-1] of the interpolant at ``(pressure, temperature)``."""
        return self._ev('s', pressure, temperature, 1, 0)

    def _ev(self, field, pressure, temperature, dx, dy):
        """``field`` or its first derivative in P (``dx``) or T (``dy``), in linear units."""
        lp, lt, shape = self._logs(pressure, temperature)
        scale = (10.0**lp) ** dx * (10.0**lt) ** dy * np.log(10.0) ** (dx + dy)
        out = (self._splines[field].ev(lp, lt, dx=dx, dy=dy) / scale).reshape(shape)
        return float(out) if out.ndim == 0 else out
