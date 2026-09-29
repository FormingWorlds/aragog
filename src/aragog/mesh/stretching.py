"""Node distributions refined towards the mantle boundaries.

A computational coordinate ``xi`` is uniform in [0, 1]; the stretched unit grid
``s(xi)`` sets the node positions. With a target only at one end the map is the
one-sided hyperbolic tangent, ``s = tanh(beta xi) / tanh(beta)`` for the outer end
(and its mirror for the inner end); with both targets it is the two-sided form of
Vinokur (1983), ``s = u / (A + (1 - A) u)``,
``u = 0.5 (1 + tanh(delta (xi - 1/2)) / tanh(delta / 2))``, with ``delta`` and ``A``
solved so that the first and last cells match their targets.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, fsolve


def _one_sided_outer(xi: npt.NDArray, last: float) -> npt.NDArray:
    h = xi[1] - xi[0]

    def cell(beta: float) -> float:
        return 1.0 - np.tanh(beta * (1.0 - h)) / np.tanh(beta) - last

    beta = brentq(cell, 1e-8, 200.0, xtol=1e-14, rtol=1e-14)
    return np.tanh(beta * xi) / np.tanh(beta)


def _two_sided(xi: npt.NDArray, first: float, last: float) -> npt.NDArray:
    h = xi[1] - xi[0]

    def grid(delta: float, A: float) -> npt.NDArray:
        u = 0.5 * (1.0 + np.tanh(delta * (xi - 0.5)) / np.tanh(0.5 * delta))
        return u / (A + (1.0 - A) * u)

    def residual(p):
        delta, logA = p
        s = grid(delta, np.exp(logA))
        return [np.log(s[1] / first), np.log((1.0 - s[-2]) / last)]

    # Vinokur's start: sinh(delta) / delta = h / sqrt(first last), A = sqrt(last / first)
    b = h / np.sqrt(first * last)
    delta0 = brentq(lambda d: np.sinh(d) / d - b, 1e-8, 50.0) if b > 1.0 else 1e-3
    sol = fsolve(residual, [delta0, 0.5 * np.log(last / first)], xtol=1e-12, full_output=True)[
        0
    ]
    s = grid(sol[0], np.exp(sol[1]))
    worst = np.max(np.abs(residual(sol)))
    if not (worst < 1e-9 and np.all(np.diff(s) > 0.0)):
        raise ValueError(
            f'two-sided mesh stretching did not converge (log residual {worst:.3g})'
        )
    return s


def stretched_unit_grid(n: int, first: float = 0.0, last: float = 0.0) -> npt.NDArray:
    """Monotone grid of ``n`` nodes on [0, 1] with prescribed end cells.

    Parameters
    ----------
    n : int
        Number of nodes (at least 3).
    first, last : float
        Thickness of the first (inner) and last (outer) cell as a fraction of the
        span; 0 leaves that end free. A non-zero value must be below the uniform
        cell ``1 / (n - 1)``.

    Returns
    -------
    ndarray
        Node positions, ``s[0] = 0`` and ``s[-1] = 1`` exactly, strictly increasing.

    Raises
    ------
    ValueError
        For a target that is negative or not below the uniform cell, or a
        two-sided solve that does not converge to a monotone grid.
    """
    xi = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    for name, val in (('first', first), ('last', last)):
        if not (np.isfinite(val) and 0.0 <= val < h):
            raise ValueError(f'{name} cell fraction must be in [0, {h:.6g}), got {val}')
    if first == 0.0 and last == 0.0:
        return xi
    if first == 0.0:
        s = _one_sided_outer(xi, last)
    elif last == 0.0:
        s = 1.0 - _one_sided_outer(xi, first)[::-1]
    else:
        s = _two_sided(xi, first, last)
    s[0], s[-1] = 0.0, 1.0
    return s
