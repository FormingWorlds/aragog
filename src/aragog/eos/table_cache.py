"""Disk cache for tabular EOS data files.

Provides atomic caching of parsed ASCII tabular data into compressed
or uncompressed NumPy binary (.npz) archives with integrity hashing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


def _compute_source_digest(filepath: Path) -> str:
    """Compute blake2b cryptographic digest of file contents.

    Parameters
    ----------
    filepath : Path
        Target file to digest.

    Returns
    -------
    str
        Hexadecimal blake2b digest string.
    """
    hasher = hashlib.blake2b()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_cache_path(
    filepath: Path,
    skiprows: int,
    dtype_str: str,
    usecols_str: str,
) -> Path | None:
    """Resolve destination path for table cache file.

    Parameters
    ----------
    filepath : Path
        Resolved path to the source ASCII table.
    skiprows : int
        Number of leading rows skipped during parsing.
    dtype_str : str
        String representation of target numpy dtype.
    usecols_str : str
        String representation of column selection.

    Returns
    -------
    Path or None
        Target path for the cache file, or None if cache directory creation fails.
    """
    custom_cache = os.environ.get('ARAGOG_TABLE_CACHE_DIR')
    fwl_data = os.environ.get('FWL_DATA')

    is_under_fwl = False
    if fwl_data:
        try:
            fwl_resolved = Path(fwl_data).resolve()
            for p in (filepath.resolve(), Path(filepath).absolute()):
                try:
                    p.relative_to(fwl_resolved)
                    is_under_fwl = True
                    break
                except (ValueError, RuntimeError):
                    continue
        except (RuntimeError, OSError):
            is_under_fwl = False

    if custom_cache:
        cache_dir = Path(custom_cache).resolve()
        use_central_root = True
    elif is_under_fwl and fwl_data:
        cache_dir = Path(fwl_data).resolve() / 'cache' / 'tables'
        use_central_root = True
    else:
        use_central_root = False

    if use_central_root:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            logger.warning(
                'Could not create table cache root %s: %s; parsing without cache',
                cache_dir,
                exc,
            )
            return None
        key_str = (
            f'{filepath.resolve()}::skiprows={skiprows}'
            f'::dtype={dtype_str}::usecols={usecols_str}'
        )
        digest = hashlib.blake2b(key_str.encode('utf-8')).hexdigest()
        return cache_dir / f'{digest}.npz'

    resolved = filepath.resolve()
    if resolved.suffix == '.npz':
        return resolved.with_name(f'{resolved.name}.cache.npz')
    return resolved.with_suffix('.npz')


def _write_cache_file(
    cache_path: Path,
    data: np.ndarray,
    source_size: int,
    source_digest: str,
    skiprows: int,
    dtype_str: str,
    usecols_str: str,
) -> None:
    """Write parsed array and validation metadata atomically to disk.

    Parameters
    ----------
    cache_path : Path
        Target cache file location.
    data : np.ndarray
        Parsed tabular array.
    source_size : int
        Size in bytes of the source ASCII file.
    source_digest : str
        Cryptographic digest of source ASCII file.
    skiprows : int
        Rows skipped in ASCII source.
    dtype_str : str
        String representation of array dtype.
    usecols_str : str
        String representation of selected columns.
    """
    try:
        cache_dir = cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=f'.{cache_path.stem}_',
            suffix='.npz',
            delete=False,
        ) as tf:
            temp_path = Path(tf.name)
        try:
            np.savez(
                temp_path,
                data=data,
                format_version=np.int64(CACHE_FORMAT_VERSION),
                source_size=np.int64(source_size),
                source_digest=np.array(source_digest),
                skiprows=np.int64(skiprows),
                dtype=np.array(dtype_str),
                usecols=np.array(usecols_str),
            )
            try:
                current_umask = os.umask(0)
                os.umask(current_umask)
                os.chmod(temp_path, 0o666 & ~current_umask)
            except OSError:
                pass
            os.replace(temp_path, cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
    except (OSError, PermissionError) as exc:
        logger.debug('Failed to write table cache %s: %s', cache_path, exc)


def read_cached_table(
    filepath: Path | str,
    skiprows: int = 0,
    skip_header: int | None = None,
    dtype: Any = float,
    usecols: Any = None,
) -> np.ndarray:
    """Load tabular data from disk using an atomic binary cache when valid.

    Parameters
    ----------
    filepath : Path or str
        Path to ASCII tabular file.
    skiprows : int, optional
        Number of leading lines to skip, default is 0.
    skip_header : int or None, optional
        Alias for skiprows matching np.genfromtxt convention.
    dtype : Any, optional
        Target numpy data type, default is float.
    usecols : Any, optional
        Columns to read from table, default is all columns.

    Returns
    -------
    np.ndarray
        Loaded data array matching the ASCII parse result.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist or is not a regular file.
    """
    fp = Path(filepath)
    if not fp.is_file():
        raise FileNotFoundError(f'Table file not found: {fp}')

    if skip_header is not None:
        skiprows = skip_header

    if dtype is None:
        dtype_str = 'None'
    else:
        dtype_str = str(np.dtype(dtype))
    usecols_str = str(usecols) if usecols is not None else ''

    source_size = fp.stat().st_size
    source_digest = _compute_source_digest(fp)

    cache_path = _resolve_cache_path(fp, skiprows, dtype_str, usecols_str)
    if cache_path is not None and cache_path.is_file():
        try:
            with np.load(cache_path) as npz:
                if (
                    int(npz['format_version']) == CACHE_FORMAT_VERSION
                    and int(npz['source_size']) == source_size
                    and str(npz['source_digest']) == source_digest
                    and int(npz['skiprows']) == skiprows
                    and str(npz['dtype']) == dtype_str
                    and str(npz['usecols']) == usecols_str
                ):
                    return np.array(npz['data'])
        except (
            KeyError,
            ValueError,
            TypeError,
            zipfile.BadZipFile,
            EOFError,
            OSError,
        ) as exc:
            logger.debug(
                'Table cache load failed for %s (%s); falling back to parse',
                cache_path,
                exc,
            )

    try:
        data = np.loadtxt(fp, skiprows=skiprows, dtype=dtype, usecols=usecols)
    except Exception:
        data = np.genfromtxt(
            fp,
            skip_header=skiprows,
            dtype=dtype,
            usecols=usecols,
        )

    if cache_path is not None:
        try:
            _write_cache_file(
                cache_path,
                data,
                source_size,
                source_digest,
                skiprows,
                dtype_str,
                usecols_str,
            )
        except Exception as exc:
            logger.debug('Failed to write table cache %s: %s', cache_path, exc)

    return data
