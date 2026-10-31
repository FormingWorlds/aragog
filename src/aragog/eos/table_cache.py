"""Disk cache for tabular EOS data files.

Provides atomic caching of parsed ASCII tabular data into uncompressed
NumPy binary (.npz) archives with integrity hashing.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import uuid
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger('fwl.' + __name__)

CACHE_FORMAT_VERSION = 2
_FAILED_CACHE_ROOTS: set[Path] = set()


def _resolve_cache_path(filepath: Path, skiprows: int) -> Path:
    """Resolve destination path for table cache file.

    Parameters
    ----------
    filepath : Path
        Resolved path to the source ASCII table.
    skiprows : int
        Number of leading rows skipped during parsing.

    Returns
    -------
    Path
        Target path for the cache file.
    """
    custom_cache = os.environ.get('ARAGOG_TABLE_CACHE_DIR')
    fwl_data = os.environ.get('FWL_DATA')

    cache_path = None
    if custom_cache:
        cache_dir = Path(custom_cache).resolve()
        key_str = f'{filepath.resolve()}::skiprows={skiprows}'
        digest = hashlib.blake2b(key_str.encode('utf-8')).hexdigest()
        cache_path = cache_dir / f'{digest}.npz'
    elif fwl_data:
        try:
            fwl_resolved = Path(fwl_data).resolve()
            if any(
                p.is_relative_to(fwl_resolved)
                for p in (filepath.resolve(), filepath.absolute())
            ):
                cache_dir = fwl_resolved / 'cache' / 'tables'
                key_str = f'{filepath.resolve()}::skiprows={skiprows}'
                digest = hashlib.blake2b(key_str.encode('utf-8')).hexdigest()
                cache_path = cache_dir / f'{digest}.npz'
        except (RuntimeError, OSError):
            pass

    if cache_path is None:
        resolved = filepath.resolve()
        cache_dir = resolved.parent
        cache_path = cache_dir / f'{resolved.name}.cache.npz'

    if cache_dir in _FAILED_CACHE_ROOTS:
        logger.debug(
            'Table cache root %s previously failed; skipping directory creation',
            cache_dir,
        )
    elif custom_cache or (fwl_data and cache_dir != filepath.resolve().parent):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _FAILED_CACHE_ROOTS.add(cache_dir)
            logger.warning(
                'Could not create table cache root %s: %s; future writes will be skipped',
                cache_dir,
                exc,
            )

    return cache_path


def _write_cache_file(
    cache_path: Path, data: np.ndarray, source_digest: str, skiprows: int
) -> None:
    """Write parsed array and validation metadata atomically to disk.

    Parameters
    ----------
    cache_path : Path
        Target cache file location.
    data : np.ndarray
        Parsed tabular array.
    source_digest : str
        Cryptographic digest of source ASCII file.
    skiprows : int
        Rows skipped in ASCII source.
    """
    temp_path = cache_path.parent / f'.{cache_path.stem}_{uuid.uuid4().hex}.npz'
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0)
    fd = os.open(temp_path, flags, 0o666)
    try:
        with open(fd, 'wb') as f:
            np.savez(
                f,
                data=data,
                format_version=np.int64(CACHE_FORMAT_VERSION),
                source_digest=np.array(source_digest),
                skiprows=np.int64(skiprows),
            )
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_cached_table(filepath: Path | str, skiprows: int = 0) -> np.ndarray:
    """Load tabular data from disk using an atomic binary cache when valid.

    Parameters
    ----------
    filepath : Path or str
        Path to ASCII tabular file.
    skiprows : int, optional
        Number of leading lines to skip, default is 0.

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

    raw = fp.read_bytes()
    source_digest = hashlib.blake2b(raw).hexdigest()

    cache_path = _resolve_cache_path(fp, skiprows)
    if cache_path.is_file():
        try:
            with np.load(cache_path) as npz:
                if (
                    int(npz['format_version']) == CACHE_FORMAT_VERSION
                    and str(npz['source_digest']) == source_digest
                    and int(npz['skiprows']) == skiprows
                ):
                    arr = npz['data']
                    if arr.size > 0:
                        return arr
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

    stream = io.TextIOWrapper(io.BytesIO(raw), newline=None)
    data = np.genfromtxt(stream, skip_header=skiprows)

    if data.size > 0 and cache_path.parent not in _FAILED_CACHE_ROOTS:
        try:
            _write_cache_file(cache_path, data, source_digest, skiprows)
        except OSError as exc:
            parent_dir = cache_path.parent
            _FAILED_CACHE_ROOTS.add(parent_dir)
            logger.warning(
                'Could not write table cache in %s: %s; future writes will be skipped',
                parent_dir,
                exc,
            )

    return data
