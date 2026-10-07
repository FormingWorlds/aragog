"""Unit tests for the EOS tabular data disk cache.

Verifies bit-equality, hash invalidation, argument tracking, error resilience,
concurrency safety, and dataset directory preservation.
"""

from __future__ import annotations

import logging
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

        temp_files = list(ro_dir.glob('.*.npz'))
        assert not temp_files, f'Temp files left in {ro_dir}: {temp_files}'

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

        temp_files = list(tmp_path.glob('.*.npz'))
        assert not temp_files, f'Temp files left in {tmp_path}: {temp_files}'

    def test_cache_write_deterministic_atomicity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify cache write uses a temporary file and replaces atomically."""
        table_file = tmp_path / 'atomic_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')
        expected_cache = table_file.with_name(f'{table_file.name}.cache.npz')

        savez_calls: list[tuple[Path, bool]] = []
        real_savez = np.savez

        def recording_savez(file, *args, **kwargs):
            savez_calls.append((Path(file), expected_cache.exists()))
            return real_savez(file, *args, **kwargs)

        replace_calls: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def recording_replace(src, dst):
            replace_calls.append((Path(src), Path(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr('aragog.eos.table_cache.np.savez', recording_savez)
        monkeypatch.setattr('aragog.eos.table_cache.os.replace', recording_replace)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert len(savez_calls) == 1
        written_path, target_existed_during_save = savez_calls[0]

        assert written_path != expected_cache
        assert not target_existed_during_save
        assert written_path.parent == expected_cache.parent
        assert written_path.name.startswith(f'.{expected_cache.stem}_')
        assert written_path.name.endswith('.npz')

        assert len(replace_calls) == 1
        src_path, dst_path = replace_calls[0]
        assert src_path == written_path
        assert dst_path == expected_cache
        assert expected_cache.is_file()

        temp_files = list(tmp_path.glob('.*.npz'))
        assert not temp_files, f'Temp files left in {tmp_path}: {temp_files}'

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

    def test_custom_cache_dir_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ARAGOG_TABLE_CACHE_DIR routes cache file to custom dir only."""
        custom_dir = tmp_path / 'custom_cache'
        custom_dir.mkdir()
        monkeypatch.setenv('ARAGOG_TABLE_CACHE_DIR', str(custom_dir))

        table_dir = tmp_path / 'tables'
        table_dir.mkdir()
        table_file = table_dir / 'test.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

        cached_files = list(custom_dir.glob('*.npz'))
        assert len(cached_files) == 1
        assert not (table_dir / f'{table_file.name}.cache.npz').exists()

    def test_cache_root_mkdir_failure_logs_warning_and_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify warning logged, no cache written, and data matches np.loadtxt when mkdir fails."""
        custom_dir = tmp_path / 'uncreatable_dir'
        monkeypatch.setenv('ARAGOG_TABLE_CACHE_DIR', str(custom_dir))

        table_file = tmp_path / 'mkdir_fail.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        orig_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            if str(self) == str(custom_dir.resolve()):
                raise OSError('Permission denied: cannot create directory')
            return orig_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'mkdir', failing_mkdir)

        with caplog.at_level(logging.WARNING):
            arr = read_cached_table(table_file)

        assert np.array_equal(arr, np.loadtxt(table_file))
        assert not custom_dir.exists()
        assert not table_file.with_name(f'{table_file.name}.cache.npz').exists()
        assert 'Could not create table cache root' in caplog.text

    def test_write_cache_file_oserror_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify reader returns parsed data cleanly when _write_cache_file raises OSError."""
        table_file = tmp_path / 'write_fail.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        def failing_write(*args, **kwargs):
            raise OSError('Disk write error')

        monkeypatch.setattr('aragog.eos.table_cache._write_cache_file', failing_write)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.loadtxt(table_file))
        assert not table_file.with_name(f'{table_file.name}.cache.npz').exists()

    def test_chmod_failure_still_gives_valid_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify os.chmod failure in _write_cache_file is handled and cache remains valid."""
        table_file = tmp_path / 'chmod_fail.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        orig_chmod = os.chmod

        def failing_chmod(path, mode):
            if 'tmp' in str(path) or 'chmod_fail' in str(path):
                raise OSError('Operation not permitted on chmod')
            return orig_chmod(path, mode)

        monkeypatch.setattr(os, 'chmod', failing_chmod)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
        arr_warm = read_cached_table(table_file)
        assert np.array_equal(arr, arr_warm)

    def test_cache_temp_file_cleanup_on_replace_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify temporary cache file is cleaned up if os.replace fails."""
        table_file = tmp_path / 'replace_fail.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        orig_replace = os.replace

        def failing_replace(src, dst):
            if 'replace_fail' in str(dst) or 'table' in str(src):
                raise OSError('Simulated replace failure')
            return orig_replace(src, dst)

        monkeypatch.setattr(os, 'replace', failing_replace)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert not table_file.with_name(f'{table_file.name}.cache.npz').exists()

    def test_missing_source_raises_filenotfound(self, tmp_path: Path) -> None:
        """Verify non-existent table file raises FileNotFoundError."""
        missing = tmp_path / 'does_not_exist.dat'
        with pytest.raises(FileNotFoundError, match='Table file not found'):
            read_cached_table(missing)

    def test_skip_header_argument(self, tmp_path: Path) -> None:
        """Verify skip_header alias correctly overrides skiprows."""
        table_file = tmp_path / 'skip_header.dat'
        table_file.write_text('# header line\n10.0 20.0\n30.0 40.0\n')
        arr = read_cached_table(table_file, skip_header=1)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 10.0

    def test_dtype_none_argument(self, tmp_path: Path) -> None:
        """Verify dtype=None argument parses and caches correctly."""
        table_file = tmp_path / 'dtype_none.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')
        arr = read_cached_table(table_file, dtype=None)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
        arr_warm = read_cached_table(table_file, dtype=None)
        assert np.array_equal(arr, arr_warm)

    def test_loadtxt_valueerror_falls_back_to_genfromtxt(self, tmp_path: Path) -> None:
        """Verify ValueError in np.loadtxt falls back to np.genfromtxt and matches."""
        table_file = tmp_path / 'missing_val.dat'
        table_file.write_text('1.0 2.0\nNA 4.0\n')

        with pytest.raises(ValueError):
            np.loadtxt(table_file)

        expected = np.genfromtxt(table_file)
        arr = read_cached_table(table_file)
        assert np.allclose(arr, expected, equal_nan=True)

        arr_warm = read_cached_table(table_file)
        assert np.allclose(arr_warm, expected, equal_nan=True)

    def test_fwl_data_resolution_error_falls_back_beside_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify resolution error on FWL_DATA falls back to placing cache beside source."""
        monkeypatch.setenv('FWL_DATA', '/nonexistent/fake/fwl/path')
        table_file = tmp_path / 'fwl_err_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        orig_resolve = Path.resolve

        def failing_resolve(self, *args, **kwargs):
            if 'fake/fwl' in str(self):
                raise OSError('Path resolution failure on FWL_DATA')
            return orig_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'resolve', failing_resolve)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
