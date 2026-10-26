"""Shared non-test helpers for solid-state convection verification tools."""

from __future__ import annotations

import contextlib
import copy
import functools
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.solver import entropy_solver as es_mod
from aragog.solver import entropy_state as estate_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX40_RE = re.compile(r'^[0-9a-f]{40}$')


def _is_valid_commit_sha(sha: object) -> bool:
    """Return True if ``sha`` is a 40-character lowercase hexadecimal commit SHA."""
    return isinstance(sha, str) and bool(_HEX40_RE.fullmatch(sha))


def _is_commit_on_origin(sha: str, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if ``sha`` is a 40-hex git commit reachable from any ``origin/*`` branch."""
    if not _is_valid_commit_sha(sha):
        return False
    try:
        obj_type = subprocess.check_output(
            ['git', '-C', str(repo_root), 'cat-file', '-t', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if obj_type != 'commit':
            return False
        branches_out = subprocess.check_output(
            ['git', '-C', str(repo_root), 'branch', '-r', '--contains', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return any(line.strip().startswith('origin/') for line in branches_out.splitlines())


def _resolve_origin_head_commit(repo_root: Path = REPO_ROOT) -> str:
    """Return ``HEAD`` SHA only when the tracked tree is clean and ``HEAD`` is on ``origin``."""
    status_out = subprocess.check_output(
        ['git', '-C', str(repo_root), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if status_out:
        raise RuntimeError(
            'Working tree has uncommitted tracked changes; commit and push before generating benchmarks'
        )
    head_sha = subprocess.check_output(
        ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if not _is_commit_on_origin(head_sha, repo_root):
        raise RuntimeError(
            f'HEAD {head_sha} is not reachable from any origin/* branch; push before generating benchmarks'
        )
    return head_sha


@functools.cache
def _parse(eos_class: type, eos_dir: str):
    try:
        return eos_class(eos_dir), None, None
    except Exception as exc:
        return None, exc, exc.__traceback__


def _parsed(eos_class: type, eos_dir: str):
    """Return the cached parse of ``eos_dir``; a failed parse re-raises without a retry."""
    eos, exc, traceback = _parse(eos_class, eos_dir)
    if exc is not None:
        raise exc.with_traceback(traceback)
    return eos


def entropy_eos_copy(
    eos_dir: Path | str,
    strict_range: bool = False,
    identity_alpha: bool = False,
    eos_class: type = EntropyEOS,
) -> EntropyEOS:
    """Return an independent copy of ``EntropyEOS(eos_dir, strict_range)``.

    The tables are parsed once per process and each call returns a deep copy,
    which starts in the state of a fresh build; only the read-only solidus and
    liquidus interpolation functions are shared. ``identity_alpha`` drops the
    ``thermal_exp`` tables, so the expansivity comes from the identity
    ``rho cp |dT/dP_S| / T`` that the JAX EOS uses.
    """
    eos = copy.deepcopy(_parsed(eos_class, str(eos_dir)))
    eos.strict_range = strict_range
    if identity_alpha:
        for name in ('thermal_exp_solid', 'thermal_exp_melt'):
            eos._tables.pop(name, None)
        eos._has_alpha_tables = False
    return eos


def entropy_eos_jax(eos_dir: Path | str):
    """Return the ``EntropyEOS_JAX`` of ``eos_dir``, parsed once per process.

    The instance is a frozen equinox module of JAX arrays, so callers share it.
    """
    from aragog.jax.eos import EntropyEOS_JAX

    return _parsed(EntropyEOS_JAX, str(eos_dir))


@contextlib.contextmanager
def unbounded_cvode_step() -> Iterator[None]:
    """Override ``EntropySolver._solve_cvode`` with ``max_step = np.inf`` on const-property runs."""
    orig_cvode = es_mod.EntropySolver._solve_cvode

    def _unclamped_cvode(self_s: Any, *a: Any, **kw: Any) -> Any:
        if self_s.entropy_eos is None and getattr(
            self_s.parameters.phase_mixed, 'const_properties', False
        ):
            if len(a) > 5:
                a = (*a[:5], np.inf, *a[6:])
            else:
                kw['max_step'] = np.inf
        return orig_cvode(self_s, *a, **kw)

    es_mod.EntropySolver._solve_cvode = _unclamped_cvode
    try:
        yield
    finally:
        es_mod.EntropySolver._solve_cvode = orig_cvode


@contextlib.contextmanager
def half_cell_surface_flux() -> Iterator[None]:
    """Override ``EntropyState.update`` so every ``outer_boundary_condition == 5`` update sets ``_heat_flux[-1]`` by half-cell conduction.

    Sets ``self_st._heat_flux[-1] = k_surf * (t_cell - T_surf) / (dr / 2)`` after
    ``orig_update`` returns. Because the stagnant-lid closure
    (``_compute_stagnant_lid_state_core`` in ``src/aragog/rheology_lid.py``, called
    inside ``EntropyState.update`` in ``src/aragog/solver/entropy_state.py``) runs
    before this assignment, the closure reads the extrapolated surface temperature
    ``T_basic[-1]`` and unpatched flux ``_jcond[-1] + _jconv[-1]``; stored
    yield-thinning and internal-heating scaling steady states are therefore steady
    for a closure fed the extrapolated surface temperature and flux.
    """
    orig_update = estate_mod.EntropyState.update

    def _half_cell_update(
        self_st: Any, entropy: Any, time_val: Any, *a: Any, **kw: Any
    ) -> None:
        orig_update(self_st, entropy, time_val, *a, **kw)
        bc_set = self_st._evaluator.boundary_conditions._settings
        if int(bc_set.outer_boundary_condition) == 5:
            k_surf = float(np.asarray(self_st.phase_basic.thermal_conductivity()).ravel()[-1])
            t_cell = float(np.asarray(self_st.phase_staggered.temperature()).ravel()[-1])
            r_b = np.asarray(self_st._evaluator.mesh.basic.radii).ravel()
            dr_half = 0.5 * float(r_b[-1] - r_b[-2])
            self_st._heat_flux[-1] = (
                k_surf * (t_cell - float(bc_set.outer_boundary_value)) / dr_half
            )

    estate_mod.EntropyState.update = _half_cell_update
    try:
        yield
    finally:
        estate_mod.EntropyState.update = orig_update


def korenaga_ti_fixed_point(
    t_stag: np.ndarray,
    z_norm: np.ndarray,
    t_surf: float,
    dth: float,
    *,
    max_iter: int = 100,
    tol_k: float = 1.0e-9,
) -> float:
    """Solve Korenaga (2009) Eq. 20 fixed point for sub-lid interior temperature ``T_i``."""
    order = np.argsort(z_norm)
    z_s, t_s_arr = z_norm[order], t_stag[order]
    ti = float(t_s_arr.max())
    for _ in range(max_iter):
        dl = (ti - t_surf) / dth
        z_above = z_s[z_s > dl]
        if z_above.size == 0:
            raise RuntimeError(
                'Korenaga fixed-point iteration did not converge: lid exceeds domain'
            )
        zz = np.concatenate([[dl], z_above])
        tt = np.interp(zz, z_s, t_s_arr)
        ti_new = float(np.trapezoid(tt, zz) / (zz[-1] - dl))
        if abs(ti_new - ti) < tol_k:
            return ti_new
        ti = ti_new
    raise RuntimeError(
        f'Korenaga fixed-point iteration did not converge within {max_iter} steps'
    )


def ols_fit(x_cols: list[np.ndarray], y_vec: np.ndarray) -> dict[str, Any]:
    """Fit ``y = c_0 + sum_j c_j x_j`` by ordinary least squares with standard errors."""
    n = len(y_vec)
    x_mat = np.column_stack([np.ones(n), *x_cols])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ coeffs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    dof = max(n - x_mat.shape[1], 1)
    cov = (ss_res / dof) * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    if len(x_cols) == 2:
        ln_a, alpha, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return {
            'a': float(np.exp(ln_a)),
            'ln_a': ln_a,
            'alpha': alpha,
            'beta': beta,
            'std_err_ln_a': float(se[0]),
            'std_err_alpha': float(se[1]),
            'std_err_beta': float(se[2]),
            'r_squared': r2,
            'n_cases': int(n),
            'dof': int(dof),
        }
    return {
        'slope': float(coeffs[1]),
        'std_err_slope': float(se[1]),
        'intercept': float(coeffs[0]),
        'r_squared': r2,
        'n_cases': int(n),
        'dof': int(dof),
    }
