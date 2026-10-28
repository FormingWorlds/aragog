"""Shared non-test helpers for solid-state convection verification tools."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.solver import entropy_solver as es_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX40_RE = re.compile(r'^[0-9a-f]{40}$')
_HEX64_RE = re.compile(r'^[0-9a-f]{64}$')
_FINGERPRINT_SUFFIXES = ('.py', '.toml', '.cfg', '.dat')
_FINGERPRINT_EXCLUDED_NAMES = frozenset({'_version.py'})

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


def _git_tracked_files(input_paths: Sequence[str], repo_root: Path) -> set[str] | None:
    """Return the git-tracked files under ``input_paths``, or None outside a git work tree."""
    try:
        out = subprocess.run(
            ['git', '-C', str(repo_root), 'ls-files', '-z', '--', *input_paths],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {rel for rel in out.decode('utf-8').split('\0') if rel}


def compute_generator_input_fingerprint(
    input_paths: Sequence[str] = ('src/aragog',),
    repo_root: Path = REPO_ROOT,
) -> tuple[str, dict[str, str]]:
    """Compute a deterministic SHA-256 content fingerprint of ``input_paths`` under ``repo_root``.

    Directories are expanded recursively to ``.py``, ``.toml``, ``.cfg``, and ``.dat`` files
    (excluding ``__pycache__``, the install-generated ``_version.py`` and, inside a git work
    tree, untracked files). Returns ``(sha256_hex, file_sha256_by_relpath)``.
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
    tracked = _git_tracked_files(input_paths, repo_root)
    files = sorted(
        rel
        for rel in collected
        if Path(rel).name not in _FINGERPRINT_EXCLUDED_NAMES
        and (tracked is None or rel in tracked)
    )
    hasher = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for rel_posix in files:
        file_bytes = (repo_root / rel_posix).read_bytes()
        file_hashes[rel_posix] = hashlib.sha256(file_bytes).hexdigest()
        hasher.update(f'{rel_posix}\0{len(file_bytes)}\0'.encode('utf-8'))
        hasher.update(file_bytes)
        hasher.update(b'\0')
    return hasher.hexdigest(), file_hashes


def _check_generator_input_fingerprint(
    commit_sha: object,
    recorded_sha256: object,
    input_paths: Sequence[str] = ('src/aragog',),
    recorded_files: Mapping[str, str] | Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
    *,
    regenerate_cmd: str = 'python tools/verification/<tool>.py --full',
    strict: bool = False,
) -> list[str]:
    """Verify ``commit_sha`` is 40-hex and check ``recorded_sha256`` against the live fingerprint.

    When ``strict=False`` (``--check`` mode), a fingerprint mismatch logs a warning to
    ``stderr`` naming the changed input files and ``regenerate_cmd`` without adding a failure.
    When ``strict=True`` (generation mode), a fingerprint mismatch is returned as an error.
    """
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

    changed_files: list[str] = []
    if isinstance(recorded_files, Mapping):
        all_keys = sorted(set(recorded_files.keys()) | set(current_files.keys()))
        changed_files = [k for k in all_keys if recorded_files.get(k) != current_files.get(k)]
    elif isinstance(recorded_files, Sequence) and not isinstance(recorded_files, (str, bytes)):
        rec_set = set(str(x) for x in recorded_files)
        cur_set = set(current_files.keys())
        changed_files = sorted(rec_set ^ cur_set)

    if changed_files or str(recorded_sha256) != current_sha256:
        files_desc = (
            ', '.join(changed_files) if changed_files else ', '.join(sorted(current_files))
        )
        msg = (
            f'WARNING: Generator inputs changed since benchmark generation '
            f'(recorded sha256 {str(recorded_sha256)[:12]} != current {current_sha256[:12]}; '
            f'changed files: {files_desc}). '
            f'Regenerate with: {regenerate_cmd}'
        )
        if strict:
            errors.append(msg)
        else:
            sys.stderr.write(f'{msg}\n')
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
            kw['max_step'] = np.inf
        return orig_cvode(self_s, *a, **kw)

    es_mod.EntropySolver._solve_cvode = _unclamped_cvode
    try:
        yield
    finally:
        es_mod.EntropySolver._solve_cvode = orig_cvode


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


def write_compact_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` to ``path`` as JSON with one compact top-level block per line."""
    blocks = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f'top-level JSON keys must be str, got {type(key).__name__}')
        blocks.append(f'{json.dumps(key)}: {json.dumps(value, separators=(",", ":"))}')
    text = '{\n' + ',\n'.join(blocks) + '\n}\n' if blocks else '{}\n'
    path.write_text(text, encoding='utf-8')
