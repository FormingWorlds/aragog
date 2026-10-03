"""Fixture verification and provenance helpers for test suites.

Provides hash computation, library version inspection, and initial profile
generators for golden numerical fixtures without importing solvers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

import numpy as np


def get_package_version(package_name: str) -> str:
    """Return installed package version via importlib.metadata.

    Parameters
    ----------
    package_name : str
        Distribution name to query.

    Returns
    -------
    str
        Version string, or 'unknown' if not installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return 'unknown'


def get_sundials_version() -> str:
    """Return SUNDIALS version string from conda metadata or package metadata.

    Returns
    -------
    str
        Version string, or 'unknown' if not detected.
    """
    env_dir = Path(sys.executable).parents[1]
    conda_meta = env_dir / 'conda-meta'
    if conda_meta.exists():
        for f in conda_meta.glob('sundials-*.json'):
            parts = f.stem.split('-')
            if len(parts) >= 2:
                return parts[1]
    return get_package_version('scikits-odes-sundials')


def compute_eos_hash(eos_dir: Path) -> str:
    """Compute recursive SHA-256 hash over directory files using relative paths.

    Parameters
    ----------
    eos_dir : Path
        Root directory containing EOS data files.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    for p in sorted(eos_dir.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(eos_dir).as_posix().encode('utf-8'))
            h.update(p.read_bytes())
    return h.hexdigest()


def get_mixed_phase_mush_initial_entropy(mesh: object) -> np.ndarray:
    """Return physical non-uniform initial entropy profile for mixed_phase_mush.

    Parameters
    ----------
    mesh : object
        Mesh object carrying staggered and basic radii arrays.

    Returns
    -------
    np.ndarray
        Initial entropy array on staggered nodes [J kg^-1 K^-1].
    """
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    r_cmb = float(r_basic[0])
    r_surf = float(r_basic[-1])
    d = r_surf - r_cmb
    return 5050.0 - 100.0 * (r_stag - r_cmb) / d


def get_creep_locked_base_initial_entropy(mesh: object) -> np.ndarray:
    """Return initial entropy profile for creep locked base fixture.

    Parameters
    ----------
    mesh : object
        Mesh object carrying staggered and basic radii arrays.

    Returns
    -------
    np.ndarray
        Initial entropy array on staggered nodes [J kg^-1 K^-1].
    """
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    r_cmb = float(r_basic[0])
    r_surf = float(r_basic[-1])
    d = r_surf - r_cmb
    return 5100.0 - 2600.0 * (r_stag - r_cmb) / d


get_partly_locked_column_initial_entropy = get_creep_locked_base_initial_entropy


def get_cold_top_lid_initial_entropy(mesh: object) -> np.ndarray:
    """Return initial entropy profile for cold top lid fixture.

    Parameters
    ----------
    mesh : object
        Mesh object carrying staggered and basic radii arrays.

    Returns
    -------
    np.ndarray
        Initial entropy array on staggered nodes [J kg^-1 K^-1].
    """
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    r_cmb = float(r_basic[0])
    r_surf = float(r_basic[-1])
    d = r_surf - r_cmb
    x = (r_stag - r_cmb) / d
    r_top_base = r_surf - 150000.0
    s_int = 4200.0 - 500.0 * x
    s_surf = 150.0
    w_surf_blend = 0.5 * (1.0 + np.tanh((r_stag - r_top_base) / 30000.0))
    return (1.0 - w_surf_blend) * s_int + w_surf_blend * s_surf
