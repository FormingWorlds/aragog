"""Unit tests for the EOS tabular data disk cache.

Verifies bit-equality, hash invalidation, argument tracking, error resilience,
concurrency safety, and dataset directory preservation.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.table_cache import (
    read_cached_table,
)


def _worker_read(args: tuple[Path, int]) -> np.ndarray:
    filepath, skiprows = args
    return read_cached_table(filepath, skiprows=skiprows)


class TestTableCache:
    """Test suite for tabular caching reader."""

    def test_cache_hit_bit_equal_float(self, tmp_path: Path) -> None:
        """Verify warm cache read produces bit-equal array and dtype."""
        table_file = tmp_path / 'test_table.dat'
        table_file.write_text('# header 1\n# header 2\n1.0 2.0 3.0\n4.0 5.0 6.0\n')

        # Cold parse
        arr_cold = read_cached_table(table_file, skiprows=2)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        # Warm hit
        arr_warm = read_cached_table(table_file, skiprows=2)
        assert np.array_equal(arr_cold, arr_warm)
        assert arr_cold.dtype == arr_warm.dtype

    def test_cache_hit_structured_dtype(self, tmp_path: Path) -> None:
        """Verify cache supports structured numpy dtypes bit-equally."""
        table_file = tmp_path / 'structured_table.dat'
        table_file.write_text('10.5 42\n20.5 84\n')

        dtype = np.dtype([('val', np.float64), ('idx', np.int32)])
        arr_cold = read_cached_table(table_file, dtype=dtype)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        arr_warm = read_cached_table(table_file, dtype=dtype)
        assert np.array_equal(arr_cold, arr_warm)
        assert arr_cold.dtype == arr_warm.dtype

    def test_cache_invalidates_on_edited_file_same_size_and_mtime(self, tmp_path: Path) -> None:
        """Verify digest-based cache invalidation when size and mtime are restored."""
        table_file = tmp_path / 'edited_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        stat_orig = table_file.stat()
        arr1 = read_cached_table(table_file)
        assert arr1[1, 0] == 3.0

        # Edit with same byte length and restore timestamp
        table_file.write_text('1.0 2.0\n5.0 4.0\n')
        assert table_file.stat().st_size == stat_orig.st_size
        os.utime(table_file, (stat_orig.st_atime, stat_orig.st_mtime))

        arr2 = read_cached_table(table_file)
        assert arr2[1, 0] == 5.0

    def test_cache_miss_on_changed_reader_args(self, tmp_path: Path) -> None:
        """Verify changed reader arguments invalidate cache."""
        table_file = tmp_path / 'args_table.dat'
        table_file.write_text('1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n')

        arr_skip0 = read_cached_table(table_file, skiprows=0)
        assert arr_skip0.shape == (3, 3)

        arr_skip1 = read_cached_table(table_file, skiprows=1)
        assert arr_skip1.shape == (2, 3)

        arr_skip2 = read_cached_table(table_file, skiprows=2)
        assert arr_skip2.shape == (3,)

        arr_f32 = read_cached_table(table_file, dtype=np.float32)
        assert arr_f32.dtype == np.float32

    def test_read_only_dir_fallback(self, tmp_path: Path) -> None:
        """Verify parser falls back cleanly if cache cannot be written."""
        ro_dir = tmp_path / 'ro_dir'
        ro_dir.mkdir()
        table_file = ro_dir / 'ro_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        # Make directory read-only
        ro_dir.chmod(0o555)
        try:
            arr = read_cached_table(table_file)
            assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
            cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
            assert not cache_file.exists()
        finally:
            ro_dir.chmod(0o755)

    def test_corrupt_or_truncated_cache_recovery(self, tmp_path: Path) -> None:
        """Verify recovery and rewrite when cache file is corrupt or truncated."""
        table_file = tmp_path / 'corrupt_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        # Create valid cache
        arr_orig = read_cached_table(table_file)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        # Truncate cache to invalid bytes
        cache_file.write_bytes(b'PK\x03\x04truncated')

        # Reading should recover and recreate valid cache
        arr_recov = read_cached_table(table_file)
        assert np.array_equal(arr_orig, arr_recov)

        # Confirm new cache is valid
        with np.load(cache_file) as npz:
            assert np.array_equal(npz['data'], arr_orig)

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """Verify concurrent worker processes safely create a valid cache."""
        table_file = tmp_path / 'concurrent_table.dat'
        lines = [f'{i}.0 {i * 2}.0\n' for i in range(100)]
        table_file.write_text(''.join(lines))

        with ProcessPoolExecutor(max_workers=4) as executor:
            tasks = [(table_file, 0) for _ in range(4)]
            results = list(executor.map(_worker_read, tasks))

        for res in results:
            assert np.array_equal(res, results[0])

        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
        with np.load(cache_file) as npz:
            assert np.array_equal(npz['data'], results[0])

    def test_fwl_data_dataset_dir_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify source directory under FWL_DATA is untouched and cache is placed centrally."""
        fwl_root = tmp_path / 'fwl_root'
        dataset_dir = fwl_root / 'datasets' / 'test_dataset'
        dataset_dir.mkdir(parents=True)
        monkeypatch.setenv('FWL_DATA', str(fwl_root))

        table_file = dataset_dir / 'source_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        # Snapshot dataset directory before read
        before_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

        # Snapshot dataset directory after read
        after_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )
        assert before_entries == after_entries

        # Verify cache written under cache/tables
        cache_tables_dir = fwl_root / 'cache' / 'tables'
        assert cache_tables_dir.is_dir()
        cached_files = list(cache_tables_dir.glob('*.npz'))
        assert len(cached_files) == 1

    def test_source_ending_in_npz_not_clobbered(self, tmp_path: Path) -> None:
        """Verify reading a file ending in .npz does not overwrite the source."""
        table_file = tmp_path / 'table.npz'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')
        orig_text = table_file.read_text()

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert table_file.read_text() == orig_text

    def test_cache_permissions_respect_umask(self, tmp_path: Path) -> None:
        """Verify cache file permissions are open to group and world according to umask."""
        table_file = tmp_path / 'table_perm.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        read_cached_table(table_file)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
        mode = cache_file.stat().st_mode & 0o777
        assert mode & 0o444 == 0o444

    def test_corrupt_type_error_metadata_fallback(self, tmp_path: Path) -> None:
        """Verify fallback when metadata contains malformed non-scalar types."""
        table_file = tmp_path / 'type_err_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        arr_orig = read_cached_table(table_file)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        # Write non-scalar array for format_version
        with np.load(cache_file) as npz:
            meta = {k: npz[k] for k in npz.files}
        meta['format_version'] = np.array([1, 2])
        np.savez(cache_file, **meta)

        arr_recov = read_cached_table(table_file)
        assert np.array_equal(arr_orig, arr_recov)
