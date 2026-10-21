"""Thermodynamic consistency residuals of the SPIDER P-S tables of one phase.

Each identity is written as a relative residual that is 0 for an exact equation of state. The
derivatives are central differences on the table's own (P, S) coordinates with a step of k grid
cells; a stencil counts only when all its cells hold numbers.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping

import numpy as np

IDENTITIES = ('dTdS', 'dTdP', 'maxwell', 'adiabat')
QUANTITIES = ('temperature', 'density', 'heat_capacity', 'thermal_exp', 'adiabat_temp_grad')
EPS_FLOOR = 5e-9  # 9 significant digits, the precision of the shipped tables
AXIS_FIT = 1e-6  # an axis is regular within AXIS_FIT steps plus eps |x| of its fit
FACTOR = 2.0  # multiple of the h^4 term; a fault 2 cells away is found only below 3


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


def _grid(tables):
    """Return the (P, S) axes after checking that every table is there and on that one grid."""
    if not isinstance(tables, Mapping):
        raise ValueError('tables is not a mapping')
    for name in QUANTITIES:
        if name not in tables:
            raise ValueError(f'the {name} table is missing')
        if not isinstance(tables[name], Mapping):
            raise ValueError(f'the {name} table is not a mapping')
        for key in ('P', 'S', 'values'):
            if key not in tables[name]:
                raise ValueError(f"the {name} table has no '{key}'")
    P, S = (np.asarray(tables['temperature'][a], float) for a in ('P', 'S'))
    for name, x in (('P', P), ('S', S)):
        if x.ndim != 1 or x.size < 2 or not (np.isfinite(x).all() and np.all(np.diff(x) > 0)):
            raise ValueError(
                f'{name} must be finite and strictly increasing, with at least 2 points'
            )
    for name in QUANTITIES:
        if not (np.array_equal(tables[name]['P'], P) and np.array_equal(tables[name]['S'], S)):
            raise ValueError(f'{name} is not on the (P, S) grid of the temperature table')
        if np.shape(tables[name]['values']) != (P.size, S.size):
            raise ValueError(f'{name} values do not have the shape of the (P, S) grid')
    return P, S


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
        If tables or a table is not a mapping, a table or its P, S or values is missing, the
        tables are not on one (P, S) grid, an axis is not finite and strictly increasing with
        at least 2 points, or values do not have the shape of the grid.
    """
    P, S = _grid(tables)
    T, rho, cp, alpha, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'dTdS': derivative(T, S, 1, k) * cp / T - 1,
            'dTdP': derivative(T, P, 0, k) / grad - 1,
            'maxwell': -derivative(np.log(rho), S, 1, k) / (rho * grad) - 1,
            'adiabat': grad * rho * cp / (alpha * T) - 1,
        }


def _axis_error(x, eps):
    """Rounding error of each stored coordinate: its departure from the fit of the axis plus
    4 ulp when the axis lies within AXIS_FIT steps plus eps |x| of a uniform fit, or within the
    same relative limit in ln x of a geometric fit; else eps |x| for the whole axis."""
    i = np.arange(x.size)
    tiny = 4 * np.finfo(float).eps * np.abs(x)
    lin = np.polyval(np.polyfit(i, x, 1), i)
    if (np.abs(x - lin) <= AXIS_FIT * np.abs(np.diff(x)).mean() + eps * np.abs(x)).all():
        return np.abs(x - lin) + tiny
    if (x > 0).all():
        ly = np.log(x)
        fit = np.polyval(np.polyfit(i, ly, 1), i)
        if (np.abs(ly - fit) <= AXIS_FIT * np.abs(np.diff(ly)).mean() + eps).all():
            return np.abs(x - np.exp(fit)) + tiny
    return eps * np.abs(x)


def _parts(tables, k, eps):
    """Per derivative identity (D, E, Q, q, A) with R + 1 = D / Q: E bounds the error of D per
    unit of stored precision from its two stored ends, q the relative error of Q per unit of
    stored precision, and A the relative error of D from the rounding of its two coordinates."""
    P, S = _grid(tables)
    T, rho, cp, _, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho = np.log(rho)
        one = np.where(np.isfinite(ln_rho), 1.0, np.nan)

        def d(f, ends, x, ax, Q, q):
            err = _axis_error(x, eps).reshape((-1, 1) if ax == 0 else (1, -1))
            A = _stencil(np.broadcast_to(err, f.shape), x, ax, k, 1.0)
            return derivative(f, x, ax, k), _stencil(ends, x, ax, k, 1.0), Q, q, A

        return {
            'dTdS': d(T, np.abs(T), S, 1, T / cp, 2.0),
            'dTdP': d(T, np.abs(T), P, 0, grad, 1.0),
            'maxwell': d(ln_rho, one, S, 1, -rho * grad, 2.0),
        }


def floors(tables, eps, k=1):
    """Round-off bound of each residual at step ``k`` for values stored to relative ``eps``.

    For R + 1 = D / Q the bound is (E / |Q| + |D / Q| q) eps + |D / Q| A, with E the error of
    the central difference from its two stored ends, q the relative error of Q, and A =
    (e[i+k] + e[i-k]) / (x[i+k] - x[i-k]) from the rounding error e of each coordinate: its
    departure from the fit of a regular axis (within AXIS_FIT steps plus eps |x| of a uniform
    or geometric grid), or eps |x| on every coordinate of any other axis.
    The adiabat identity, with no derivative, gets |R + 1| 5 eps from its five stored factors.

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

    Raises
    ------
    ValueError
        As for ``residuals``.
    """
    _grid(tables)
    T, rho, cp, alpha, grad = (np.asarray(tables[q]['values'], float) for q in QUANTITIES)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = {
            n: (E / np.abs(Q) + np.abs(D / Q) * q) * eps + np.abs(D / Q) * A
            for n, (D, E, Q, q, A) in _parts(tables, k, eps).items()
        }
        out['adiabat'] = np.abs(grad * rho * cp / (alpha * T)) * 5 * eps
    return out


def _eps(eps):
    """``eps`` raised to ``EPS_FLOOR``; ValueError unless it is a real number in [0, 1)."""
    if isinstance(eps, numbers.Real) and not isinstance(eps, bool) and 0 <= eps < 1:
        return max(float(eps), EPS_FLOOR)
    raise ValueError(f'eps must be a real number in [0, 1), got {eps!r}')


def check(tables, eps):
    """Test each cell against the Richardson bound of each identity.

    A derivative identity fails where |R_x(h)| > floor_x + FACTOR * est, with R_x(kh) =
    (4 R_k - R_2k) / 3, floor_x = (4 floor_1 + floor_2) / 3 and est = |R_x(2h) - R_x(h)| / 15,
    the measured h^4 term of R_x(h) at the cell; the adiabat identity fails where |R| > floor.
    The floor uses eps_eff = max(eps, EPS_FLOOR): every table is checked at the 9-digit floor
    at best, so a single value off by about 1e-8 (relative) or less is out of reach.
    The floor also covers the rounding of the stored P and S coordinates. A derivative identity
    is tested only where 4 cells on each side along its axis hold values; NaN marks a missing
    value, and a stored 0 or inf is a value. Where all values an identity reads are present but
    its residual or bound is not finite, the cell fails; an identity is skipped only when no
    cell holds all values it reads. On a grid that is not uniform the h^4 estimate holds only
    where the spacing changes smoothly. On a grid linear in P, the h^4 term of dTdP grows as
    P / dP falls: on the 199 x 201 test grid a 1e-6 temperature fault is found at all 4 cells 1
    and 2 steps away from P / dP of about 27, and at none below about 10.

    Parameters
    ----------
    tables : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored columns (0.5 x 10^(1 - n) for n significant digits).

    Returns
    -------
    dict
        Identity name -> dict with ``rx`` (the Richardson residual, defined also at untested
        cells), ``bound``, ``tested`` (both finite), ``fail`` (tested and above the bound, or
        with all values present and a residual or bound that is not finite) arrays, and
        ``skipped``, True when no cell holds all values the identity reads.

    Raises
    ------
    ValueError
        If eps is not a real number in [0, 1), a table, its P, S or values is missing, a table
        is not a mapping or off the grid, or every identity is skipped.
    """
    eps = _eps(eps)
    r = {k: residuals(tables, k) for k in (1, 2, 4)}
    ones = {
        q: {
            **tables[q],
            'values': np.where(np.isnan(np.asarray(tables[q]['values'], float)), np.nan, 1.0),
        }
        for q in QUANTITIES
    }
    present = {k: residuals(ones, k) for k in (1, 2, 4)}  # finite where every value is present
    f1, f2 = floors(tables, eps, 1), floors(tables, eps, 2)
    out = {}
    with np.errstate(invalid='ignore'):
        for n in IDENTITIES:
            if n == 'adiabat':
                rx, bound, have = r[1][n], f1[n], np.isfinite(present[1][n])
            else:
                rx = (4 * r[1][n] - r[2][n]) / 3
                est = np.abs((4 * r[2][n] - r[4][n]) / 3 - rx) / 15
                bound = (4 * f1[n] + f2[n]) / 3 + FACTOR * est
                have = np.isfinite(present[1][n] + present[2][n] + present[4][n])
            tested = np.isfinite(rx) & np.isfinite(bound)
            out[n] = {
                'rx': rx,
                'bound': bound,
                'tested': tested,
                'fail': (tested & (np.abs(rx) > bound)) | (have & ~tested),
                'skipped': not have.any(),
            }
    if all(v['skipped'] for v in out.values()):
        raise ValueError(f'no cell tested for {", ".join(out)}')
    return out
