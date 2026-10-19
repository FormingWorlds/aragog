"""Thermodynamic consistency residuals of the SPIDER P-S tables of one phase.

Each identity is written as a relative residual that is 0 for an exact equation of state. The
derivatives are central differences on the table's own (P, S) coordinates with a step of k grid
cells; a stencil counts only when all its cells hold numbers.
"""

from __future__ import annotations

import numpy as np

IDENTITIES = ('dTdS', 'dTdP', 'maxwell', 'adiabat')
QUANTITIES = ('temperature', 'density', 'heat_capacity', 'thermal_exp', 'adiabat_temp_grad')


def derivative(f, x, axis, k):
    """d f / d x along ``axis`` with a step of ``k`` cells, NaN where a stencil cell is missing.

    Parameters
    ----------
    f : ndarray
        Values on the (P, S) grid.
    x : ndarray
        Coordinates along ``axis``, increasing.
    axis : int
        0 for P, 1 for S.
    k : int
        Stencil step in grid cells.
    """
    n = f.shape[axis]
    out = np.full(f.shape, np.nan)
    if n <= 2 * k:
        return out
    run = np.cumsum(~np.isfinite(f), axis=axis)  # a missing cell between the ends changes it

    def cut(a, b):
        return tuple(slice(a, b) if ax == axis else slice(None) for ax in range(f.ndim))

    shape = [1] * f.ndim
    shape[axis] = n - 2 * k
    dx = (x[2 * k :] - x[: n - 2 * k]).reshape(shape)
    ok = np.isfinite(f[cut(0, n - 2 * k)]) & (run[cut(2 * k, n)] == run[cut(0, n - 2 * k)])
    out[cut(k, n - k)] = np.where(ok, (f[cut(2 * k, n)] - f[cut(0, n - 2 * k)]) / dx, np.nan)
    return out


def residuals(tables, k=1):
    """Relative residuals of the P-S identities at stencil step ``k`` cells.

    Parameters
    ----------
    tables : dict
        Quantity name (``QUANTITIES``) -> table as from
        ``aragog.eos.entropy._load_spider_ps_table`` (``P``, ``S`` in SI, ``values`` of shape
        (n_P, n_S)), all on one grid.
    k : int
        Stencil step in grid cells.

    Returns
    -------
    dict
        Identity name -> residual array on the grid (NaN where it cannot be formed):
        ``dTdS``: (dT/dS)_P cp / T - 1; ``dTdP``: (dT/dP)_S from T over the tabulated (dT/dP)_S
        - 1; ``maxwell``: -(dln rho/dS)_P / (rho (dT/dP)_S) - 1, from (dT/dP)_S = (dv/dS)_P;
        ``adiabat``: (dT/dP)_S rho cp / (alpha T) - 1 (no derivative).

    Raises
    ------
    ValueError
        If the tables are not on one (P, S) grid.
    """
    P, S = tables['temperature']['P'], tables['temperature']['S']
    for name in QUANTITIES:
        if not (np.array_equal(tables[name]['P'], P) and np.array_equal(tables[name]['S'], S)):
            raise ValueError(f'{name} is not on the (P, S) grid of the temperature table')
    T, rho, cp, alpha, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'dTdS': derivative(T, S, 1, k) * cp / T - 1,
            'dTdP': derivative(T, P, 0, k) / grad - 1,
            'maxwell': -derivative(np.log(rho), S, 1, k) / (rho * grad) - 1,
            'adiabat': grad * rho * cp / (alpha * T) - 1,
        }
