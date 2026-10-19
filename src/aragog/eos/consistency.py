"""Thermodynamic consistency residuals of the SPIDER P-S tables of one phase.

Each identity is written as a relative residual that is 0 for an exact equation of state. The
derivatives are central differences on the table's own (P, S) coordinates with a step of k grid
cells; a stencil counts only when all its cells hold numbers.
"""

from __future__ import annotations

import numbers

import numpy as np

IDENTITIES = ('dTdS', 'dTdP', 'maxwell', 'adiabat')
QUANTITIES = ('temperature', 'density', 'heat_capacity', 'thermal_exp', 'adiabat_temp_grad')
EPS_FLOOR = 5e-9  # 9 significant digits, the precision of the shipped tables
FACTOR = 2.0  # allowed multiple of the h^4 term; a fault 2 cells away is found only below 3


def _stencil(f, x, axis, k, sign):
    """(f[i+k] + sign f[i-k]) / (x[i+k] - x[i-k]), NaN where a stencil cell is missing."""
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
    out[cut(k, n - k)] = np.where(
        ok, (f[cut(2 * k, n)] + sign * f[cut(0, n - 2 * k)]) / dx, np.nan
    )
    return out


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
    return _stencil(f, x, axis, k, -1.0)


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
        If the tables are not on one (P, S) grid, an axis is not strictly increasing, or values
        do not have the shape of the grid.
    """
    P, S = tables['temperature']['P'], tables['temperature']['S']
    for name, x in (('P', P), ('S', S)):
        if not np.all(np.diff(x) > 0):
            raise ValueError(f'{name} must be strictly increasing')
    for name in QUANTITIES:
        if not (np.array_equal(tables[name]['P'], P) and np.array_equal(tables[name]['S'], S)):
            raise ValueError(f'{name} is not on the (P, S) grid of the temperature table')
        if np.shape(tables[name]['values']) != (len(P), len(S)):
            raise ValueError(f'{name} values do not have the shape of the (P, S) grid')
    T, rho, cp, alpha, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'dTdS': derivative(T, S, 1, k) * cp / T - 1,
            'dTdP': derivative(T, P, 0, k) / grad - 1,
            'maxwell': -derivative(np.log(rho), S, 1, k) / (rho * grad) - 1,
            'adiabat': grad * rho * cp / (alpha * T) - 1,
        }


def _parts(tables, k):
    """Per derivative identity (D, E, Q, q) with R + 1 = D / Q: E bounds the error of D per unit
    of stored precision, q the relative error of Q per unit of stored precision."""
    P, S = tables['temperature']['P'], tables['temperature']['S']
    T, rho, cp, _, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho = np.log(rho)
        one = np.where(np.isfinite(ln_rho), 1.0, np.nan)
        return {
            'dTdS': (derivative(T, S, 1, k), _stencil(np.abs(T), S, 1, k, 1.0), T / cp, 2.0),
            'dTdP': (derivative(T, P, 0, k), _stencil(np.abs(T), P, 0, k, 1.0), grad, 1.0),
            'maxwell': (
                derivative(ln_rho, S, 1, k),
                _stencil(one, S, 1, k, 1.0),
                -rho * grad,
                2.0,
            ),
        }


def floors(tables, eps, k=1):
    """Round-off bound of each residual at step ``k`` for values stored to relative ``eps``.

    For R + 1 = D / Q the bound is (E / |Q| + |D / Q| q) eps, with E the error of the central
    difference from its two stored ends and q the relative error of Q; the adiabat identity,
    with no derivative, gets |R + 1| 5 eps from its five stored factors.

    Parameters
    ----------
    tables : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored values.
    k : int
        Stencil step in grid cells.

    Returns
    -------
    dict
        Identity name -> floor array on the grid.
    """
    T, rho, cp, alpha, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = {
            n: (E / np.abs(Q) + np.abs(D / Q) * q) * eps
            for n, (D, E, Q, q) in _parts(tables, k).items()
        }
        out['adiabat'] = np.abs(grad * rho * cp / (alpha * T)) * 5 * eps
    return out


def _eps(eps):
    """``eps`` raised to ``EPS_FLOOR``; ValueError unless it is a real number in [0, 1)."""
    real = isinstance(eps, (numbers.Real, np.integer, np.floating))
    try:
        x = float(eps) if real and not isinstance(eps, (bool, np.bool_)) else np.nan
    except OverflowError:
        x = np.nan
    if not 0 <= x < 1:
        raise ValueError(f'eps must be a finite number in [0, 1), got {eps!r}')
    return max(x, EPS_FLOOR)


def check(tables, eps):
    """Test each cell against the Richardson bound of each identity.

    A derivative identity fails where |R_x(h)| > floor_x + FACTOR * est, with R_x(kh) =
    (4 R_k - R_2k) / 3, floor_x = (4 floor_1 + floor_2) / 3 and est = |R_x(2h) - R_x(h)| / 15,
    the measured h^4 term of R_x(h) at the cell; the adiabat identity fails where |R| > floor.
    The floor uses eps_eff = max(eps, EPS_FLOOR): every table is checked at the 9-digit floor
    at best, so a single value off by about 1e-8 (relative) or less is out of reach on any table.
    A derivative identity is tested only where 4 cells on each side along its axis hold values; a
    cell where a factor of an identity is 0 or not finite is not tested for it. On a grid that is
    not uniform the h^4 estimate holds only where the spacing changes smoothly.

    Parameters
    ----------
    tables : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored columns (0.5 x 10^(1 - n) for n significant digits).

    Returns
    -------
    dict
        Identity name -> dict with ``rx`` (the tested residual), ``bound``, ``tested`` (both
        finite) and ``fail`` (tested and above the bound) arrays.

    Raises
    ------
    ValueError
        If eps is not a real number in [0, 1), or an identity tests no cell.
    """
    eps = _eps(eps)
    r = {k: residuals(tables, k) for k in (1, 2, 4)}
    f1, f2 = floors(tables, eps, 1), floors(tables, eps, 2)
    out = {}
    with np.errstate(invalid='ignore'):
        for n in IDENTITIES:
            if n == 'adiabat':
                rx, bound = r[1][n], f1[n]
            else:
                rx = (4 * r[1][n] - r[2][n]) / 3
                est = np.abs((4 * r[2][n] - r[4][n]) / 3 - rx) / 15
                bound = (4 * f1[n] + f2[n]) / 3 + FACTOR * est
            tested = np.isfinite(rx) & np.isfinite(bound)
            out[n] = {
                'rx': rx,
                'bound': bound,
                'tested': tested,
                'fail': tested & (np.abs(rx) > bound),
            }
    empty = [n for n, v in out.items() if not v['tested'].any()]
    if empty:
        raise ValueError(f'no cell tested for {", ".join(empty)}')
    return out
