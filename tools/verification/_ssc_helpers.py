"""Shared non-test helpers for solid-state convection verification tools."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.solver import entropy_solver as es_mod
from aragog.solver import entropy_state as estate_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX40_RE = re.compile(r'^[0-9a-f]{40}$')
_HEX64_RE = re.compile(r'^[0-9a-f]{64}$')
_FINGERPRINT_SUFFIXES = ('.py', '.toml', '.cfg', '.dat')

CLOSURE_SCALAR_KEYS: tuple[str, ...] = (
    'd_lid',
    'd_lid_raw',
    'r_lid_base',
    'P_lid_base',
    'T_lid',
    'T_lid_iso',
    'T_i',
    'P_T_i',
    'r_i',
    'dT_rh',
    'theta',
    'eta_i',
    'mu_i',
    'eta_solid_arrhenius',
    'eta_contrast',
    'delta_rh',
    'v_i',
    'v_m',
    'Ra_eff',
    'Ra_rh',
    'tau_d',
    'tau_d_lid',
    'tau_buoy',
    'tau_d_over_tau_buoy',
    'tau_y_lid',
    'w_y',
    'w_active',
    'lid_regime',
    'lid_cell_count',
)


def _is_valid_commit_sha(sha: object) -> bool:
    """Return True if ``sha`` is a 40-character lowercase hexadecimal commit SHA."""
    return isinstance(sha, str) and bool(_HEX40_RE.fullmatch(sha))


def _is_valid_sha256(digest: object) -> bool:
    """Return True if ``digest`` is a 64-character lowercase hexadecimal SHA-256 digest."""
    return isinstance(digest, str) and bool(_HEX64_RE.fullmatch(digest))


def compute_generator_input_fingerprint(
    input_paths: Sequence[str] = ('src/aragog',),
    repo_root: Path = REPO_ROOT,
) -> tuple[str, list[str]]:
    """Compute a deterministic SHA-256 content fingerprint of ``input_paths`` under ``repo_root``.

    Directories are expanded recursively to ``.py``, ``.toml``, ``.cfg``, and ``.dat`` files
    (excluding ``__pycache__``). Returns ``(sha256_hex, sorted_posix_relpaths)``.
    """
    collected: set[str] = set()
    for rel_str in input_paths:
        target = repo_root / rel_str
        if not target.exists():
            raise FileNotFoundError(
                f'Generator input path {rel_str!r} does not exist under {repo_root}'
            )
        if target.is_file():
            collected.add(target.relative_to(repo_root).as_posix())
        elif target.is_dir():
            for child in target.rglob('*'):
                if (
                    child.is_file()
                    and '__pycache__' not in child.parts
                    and child.suffix in _FINGERPRINT_SUFFIXES
                ):
                    collected.add(child.relative_to(repo_root).as_posix())
    files = sorted(collected)
    hasher = hashlib.sha256()
    for rel_posix in files:
        file_bytes = (repo_root / rel_posix).read_bytes()
        hasher.update(f'{rel_posix}\0{len(file_bytes)}\0'.encode('utf-8'))
        hasher.update(file_bytes)
        hasher.update(b'\0')
    return hasher.hexdigest(), files


def _check_generator_input_fingerprint(
    commit_sha: object,
    recorded_sha256: object,
    input_paths: Sequence[str] = ('src/aragog',),
    recorded_files: Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Verify ``commit_sha`` is 40-hex and ``recorded_sha256`` matches the live content fingerprint."""
    errors: list[str] = []
    if not _is_valid_commit_sha(commit_sha):
        errors.append(f'Recorded commit {commit_sha!r} is not a valid 40-hex commit SHA')
    if not _is_valid_sha256(recorded_sha256):
        errors.append(
            f'Recorded generator_input_sha256 {recorded_sha256!r} is not a valid 64-hex SHA-256 digest'
        )
        return errors
    try:
        current_sha256, current_files = compute_generator_input_fingerprint(
            input_paths, repo_root=repo_root
        )
    except Exception as exc:
        errors.append(f'Failed to compute generator input fingerprint: {exc}')
        return errors
    if recorded_files is not None and list(recorded_files) != current_files:
        errors.append(
            f'Generator input file list mismatch: recorded {len(recorded_files)} files '
            f'vs current {len(current_files)} files'
        )
    if str(recorded_sha256) != current_sha256:
        errors.append(
            f'Generator inputs changed since benchmark generation: '
            f'recorded sha256 {str(recorded_sha256)[:12]} != current {current_sha256[:12]}'
        )
    return errors


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
    """Override ``EntropyState.update`` so every ``outer_boundary_condition == 5`` update sets ``_heat_flux[-1]`` by half-cell conduction."""
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


def _rel_diff_signed(new_val: float, old_val: float) -> float:
    return (new_val - old_val) / max(abs(old_val), 1.0e-30)


def evaluate_bc5_surface_patch_state(
    params: Any, s_vec: np.ndarray, t_yr: float
) -> dict[str, Any]:
    """Compare stored half-cell-patched BC 5 state against unpatched BC 5 and consistent surface closure."""
    from aragog.config.phases import _compute_stagnant_lid_state_core, compute_mlt_top_slope
    from aragog.solver.entropy_solver import EntropySolver

    with half_cell_surface_flux():
        solver_stored = EntropySolver(params, entropy_eos=None)
        solver_stored.initialize()
        rhs_stored = np.asarray(solver_stored._dSdt_single(float(t_yr), s_vec), dtype=float)
        st_stored = solver_stored.state
        lid_stored = dict(st_stored.lid_state)
        t_basic_extrap = (
            np.asarray(st_stored.phase_basic.temperature(), dtype=float).ravel().copy()
        )
        t_stag = np.asarray(st_stored.phase_staggered.temperature(), dtype=float).ravel().copy()
        jcond_unpatched = np.asarray(st_stored._jcond, dtype=float).ravel().copy()
        f_top_patched = float(st_stored._heat_flux[-1])
        eta_eff_stored = np.asarray(st_stored.eta_eff_basic, dtype=float).ravel().copy()
        eta_diff_b = np.asarray(st_stored.phase_basic.eta_diff(), dtype=float).ravel().copy()

    solver_unpatched = EntropySolver(params, entropy_eos=None)
    solver_unpatched.initialize()
    rhs_unpatched = np.asarray(solver_unpatched._dSdt_single(float(t_yr), s_vec), dtype=float)
    f_top_unpatched = float(solver_unpatched.state._heat_flux[-1])

    pass1_captured: dict[str, Any] = {}
    orig_lid_core = estate_mod._compute_stagnant_lid_state_core

    def _spy_lid_core(**kwargs: Any) -> dict[str, Any]:
        for k, v in kwargs.items():
            pass1_captured[k] = v.copy() if isinstance(v, np.ndarray) else v
        return orig_lid_core(**kwargs)

    estate_mod._compute_stagnant_lid_state_core = _spy_lid_core
    try:
        with half_cell_surface_flux():
            solver_spy = EntropySolver(params, entropy_eos=None)
            solver_spy.initialize()
            solver_spy._dSdt_single(float(t_yr), s_vec)
    finally:
        estate_mod._compute_stagnant_lid_state_core = orig_lid_core

    t_surf_bc = float(params.boundary_conditions.outer_boundary_value)
    kw_patched = dict(pass1_captured)
    t_mod = pass1_captured['temperature'].copy()
    t_mod[-1] = t_surf_bc
    f_tot_mod = pass1_captured['total_flux'].copy()
    f_tot_mod[-1] = f_top_patched
    f_conv_mod = pass1_captured['convective_flux'].copy()
    f_conv_mod[-1] = 0.0
    kw_patched['temperature'] = t_mod
    kw_patched['total_flux'] = f_tot_mod
    kw_patched['convective_flux'] = f_conv_mod
    lid_patched = _compute_stagnant_lid_state_core(**kw_patched)

    def _consistent_lid_core(**kwargs: Any) -> dict[str, Any]:
        kw = dict(kwargs)
        t_arr = np.asarray(kw['temperature'], dtype=float).copy()
        t_arr[-1] = t_surf_bc
        ft_arr = np.asarray(kw['total_flux'], dtype=float).copy()
        ft_arr[-1] = f_top_patched
        fc_arr = np.asarray(kw['convective_flux'], dtype=float).copy()
        fc_arr[-1] = 0.0
        kw['temperature'] = t_arr
        kw['total_flux'] = ft_arr
        kw['convective_flux'] = fc_arr
        return orig_lid_core(**kw)

    estate_mod._compute_stagnant_lid_state_core = _consistent_lid_core
    try:
        with half_cell_surface_flux():
            solver_c1 = EntropySolver(params, entropy_eos=None)
            solver_c1.initialize()
            rhs_c1 = np.asarray(solver_c1._dSdt_single(float(t_yr), s_vec), dtype=float)
            st_c1 = solver_c1.state
            lid_c1 = dict(st_c1.lid_state)
            eta_eff_c1 = np.asarray(st_c1.eta_eff_basic, dtype=float).ravel().copy()
    finally:
        estate_mod._compute_stagnant_lid_state_core = orig_lid_core

    k_cond = float(np.asarray(st_stored.phase_basic.thermal_conductivity()).ravel()[-1])
    rheo_p = st_stored.phase_basic._rheology
    a_lid = float(rheo_p.lid_contrast_coeff)
    ti_old = float(lid_stored['T_i'])
    ti_new = float(lid_patched['T_i'])
    th_old = float(lid_stored['theta'])
    th_new = float(lid_patched['theta'])
    dl_cond_old_m = (1.0 - a_lid / th_old) * (k_cond * (ti_old - t_surf_bc) / f_top_patched)
    dl_cond_new_m = (1.0 - a_lid / th_new) * (k_cond * (ti_new - t_surf_bc) / f_top_patched)
    dl_cond_unpatched_m = (1.0 - a_lid / th_old) * (
        k_cond * (ti_old - t_surf_bc) / f_top_unpatched
    )
    s_top_old = float(
        compute_mlt_top_slope(th_old, s_top_inf=rheo_p.s_top_inf, c_theta=rheo_p.c_theta)
    )
    s_top_new = float(
        compute_mlt_top_slope(th_new, s_top_inf=rheo_p.s_top_inf, c_theta=rheo_p.c_theta)
    )

    w_lid_old = np.asarray(lid_stored['w_lid'], dtype=float).ravel()
    w_lid_new = np.asarray(lid_patched['w_lid'], dtype=float).ravel()
    eta_ratio_old = (
        float(np.min(eta_eff_stored[w_lid_old > 0.5]) / float(lid_stored['eta_i']))
        if np.any(w_lid_old > 0.5)
        else None
    )
    eta_ratio_new = (
        float(np.min(eta_eff_c1[w_lid_new > 0.5]) / float(lid_c1['eta_i']))
        if np.any(w_lid_new > 0.5)
        else None
    )

    d_lid_old = float(lid_stored['d_lid'])
    d_lid_new = float(lid_patched['d_lid'])
    return {
        'T_surf_bc_K': t_surf_bc,
        'T_basic_extrapolated_K': float(t_basic_extrap[-1]),
        'T_stag_top_cell_K': float(t_stag[-1]),
        'with_patch_F_top_W_m2': f_top_patched,
        'shipped_bc5_F_top_W_m2': f_top_unpatched,
        'jcond_unpatched_surface_W_m2': float(jcond_unpatched[-1]),
        'with_patch_max_abs_rhs_J_kg_K_yr': float(np.max(np.abs(rhs_stored))),
        'shipped_bc5_max_abs_rhs_J_kg_K_yr': float(np.max(np.abs(rhs_unpatched))),
        'consistent_max_abs_rhs_J_kg_K_yr': float(np.max(np.abs(rhs_c1))),
        'with_patch_d_lid_cond_km': 1.0e-3 * dl_cond_old_m,
        'shipped_bc5_d_lid_cond_km': 1.0e-3 * dl_cond_unpatched_m,
        'consistent_d_lid_cond_km': 1.0e-3 * dl_cond_new_m,
        'd_lid_pass1_rel_diff': _rel_diff_signed(d_lid_new, d_lid_old),
        'd_lid_pass1_abs_rel_diff': abs(_rel_diff_signed(d_lid_new, d_lid_old)),
        'd_lid_cond_lid_rel_diff': _rel_diff_signed(dl_cond_new_m, dl_cond_old_m),
        'theta_stored': th_old,
        'theta_consistent': th_new,
        'theta_rel_diff': _rel_diff_signed(th_new, th_old),
        's_top_stored': s_top_old,
        's_top_consistent': s_top_new,
        's_top_rel_diff': _rel_diff_signed(s_top_new, s_top_old),
        'eta_eff_lid_over_eta_i_stored': eta_ratio_old,
        'eta_eff_lid_over_eta_i_consistent': eta_ratio_new,
        'eta_eff_lid_over_eta_i_rel_diff': (
            _rel_diff_signed(eta_ratio_new, eta_ratio_old)
            if (eta_ratio_old is not None and eta_ratio_new is not None)
            else None
        ),
        'n_yielding_basic_nodes_stored': int(np.sum(eta_eff_stored < 0.999 * eta_diff_b)),
        'n_yielding_basic_nodes_consistent': int(np.sum(eta_eff_c1 < 0.999 * eta_diff_b)),
    }
