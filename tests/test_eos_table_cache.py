"""Unit tests for the EOS tabular data disk cache.

Verifies bit-equality, hash invalidation, argument tracking, error resilience,
concurrency safety, and dataset directory preservation.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.table_cache import (
    _FAILED_CACHE_ROOTS,
    read_cached_table,
)

pytestmark = pytest.mark.unit


def _worker_read(args: tuple[Path, int]) -> np.ndarray:
    filepath, skiprows = args
    return read_cached_table(filepath, skiprows=skiprows)


@pytest.fixture(autouse=True)
def _clean_table_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ARAGOG_TABLE_CACHE_DIR is unset and cache roots cleared by default."""
    monkeypatch.delenv('ARAGOG_TABLE_CACHE_DIR', raising=False)
    _FAILED_CACHE_ROOTS.clear()


@pytest.fixture
def table_file(tmp_path: Path) -> Path:
    """Return a simple two-by-two test table file."""
    p = tmp_path / 'test_table.dat'
    p.write_text('1.0 2.0\n3.0 4.0\n')
    return p


class TestTableCache:
    """Test suite for tabular caching reader."""

    def test_cache_hit_bit_equal_float(self, tmp_path: Path) -> None:
        """Verify warm cache read produces bit-equal array and dtype."""
        table_file = tmp_path / 'test_table.dat'
        table_file.write_text('# header 1\n# header 2\n1.0 2.0 3.0\n4.0 5.0 6.0\n')

        arr_cold = read_cached_table(table_file, skiprows=2)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        arr_warm = read_cached_table(table_file, skiprows=2)
        assert np.array_equal(arr_cold, arr_warm)
        assert arr_cold.dtype == arr_warm.dtype

    def test_cache_invalidates_on_edited_file_same_size_and_mtime(self, tmp_path: Path) -> None:
        """Verify digest-based cache invalidation when size and mtime are restored."""
        table_file = tmp_path / 'edited_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        stat_orig = table_file.stat()
        arr1 = read_cached_table(table_file)
        assert arr1[1, 0] == 3.0

        table_file.write_text('1.0 2.0\n5.0 4.0\n')
        assert table_file.stat().st_size == stat_orig.st_size
        os.utime(table_file, (stat_orig.st_atime, stat_orig.st_mtime))

        arr2 = read_cached_table(table_file)
        assert arr2[1, 0] == 5.0

    def test_cache_miss_on_changed_reader_args(self, tmp_path: Path) -> None:
        """Verify changed skiprows arguments invalidate cache."""
        table_file = tmp_path / 'args_table.dat'
        table_file.write_text('1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n')

        arr_skip0 = read_cached_table(table_file, skiprows=0)
        assert arr_skip0.shape == (3, 3)

        arr_skip1 = read_cached_table(table_file, skiprows=1)
        assert arr_skip1.shape == (2, 3)

        arr_skip2 = read_cached_table(table_file, skiprows=2)
        assert arr_skip2.shape == (3,)

    def test_read_only_dir_fallback(self, tmp_path: Path) -> None:
        """Verify parser falls back cleanly if cache cannot be written."""
        ro_dir = tmp_path / 'ro_dir'
        ro_dir.mkdir()
        table_file = ro_dir / 'ro_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

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

    @pytest.mark.parametrize('corruption_mode', ['truncated', 'bad_metadata'])
    def test_corrupt_cache_recovery(self, table_file: Path, corruption_mode: str) -> None:
        """Verify recovery and rewrite when cache file is corrupt or has invalid metadata."""
        arr_orig = read_cached_table(table_file)
        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        if corruption_mode == 'truncated':
            cache_file.write_bytes(b'PK\x03\x04truncated')
        elif corruption_mode == 'bad_metadata':
            with np.load(cache_file) as npz:
                meta = {k: npz[k] for k in npz.files}
            meta['format_version'] = np.array([1, 2])
            np.savez(cache_file, **meta)

        arr_recov = read_cached_table(table_file)
        assert np.array_equal(arr_orig, arr_recov)

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

        savez_calls: list[bool] = []
        real_savez = np.savez

        def recording_savez(file, *args, **kwargs):
            savez_calls.append(expected_cache.exists())
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
        assert not savez_calls[0]

        assert len(replace_calls) == 1
        src_path, dst_path = replace_calls[0]
        assert src_path != expected_cache
        assert src_path.parent == expected_cache.parent
        assert src_path.name.startswith(f'.{expected_cache.stem}_')
        assert src_path.name.endswith('.npz')
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

        before_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

        after_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )
        assert before_entries == after_entries

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
        """Verify cache file permissions respect process umask."""
        for umask_val, expected_mode in [(0o077, 0o600), (0o022, 0o644)]:
            orig_umask = os.umask(umask_val)
            try:
                table_file = tmp_path / f'table_perm_{umask_val:o}.dat'
                table_file.write_text('1.0 2.0\n3.0 4.0\n')
                read_cached_table(table_file)
                cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
                assert cache_file.is_file()
                mode = cache_file.stat().st_mode & 0o777
                assert mode == expected_mode
            finally:
                os.umask(orig_umask)

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

    def test_read_only_cache_root_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify 5 reads of a table under a read-only root emit exactly 1 warning."""
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

        with caplog.at_level(logging.WARNING, logger='fwl.aragog.eos.table_cache'):
            for _ in range(5):
                arr = read_cached_table(table_file)
                assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and 'Could not create table cache root' in r.message
        ]
        assert len(warning_records) == 1
        assert not custom_dir.exists()
        assert not table_file.with_name(f'{table_file.name}.cache.npz').exists()

    def test_cache_temp_file_cleanup_on_replace_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify temporary cache file is cleaned up if os.replace fails."""
        table_file = tmp_path / 'replace_fail.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        orig_replace = os.replace

        def failing_replace(src, dst):
            if 'replace_fail' in str(dst):
                raise OSError('Simulated replace failure')
            return orig_replace(src, dst)

        monkeypatch.setattr(os, 'replace', failing_replace)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert not table_file.with_name(f'{table_file.name}.cache.npz').exists()
        leftover_temp = list(tmp_path.glob('.replace_fail*.npz'))
        assert not leftover_temp

    def test_concurrent_rewrite_during_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify modifying file on disk during parse does not corrupt cached data or digest."""
        table_file = tmp_path / 'rewrite_during_parse.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')
        orig_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        orig_digest = hashlib.blake2b(table_file.read_bytes()).hexdigest()

        orig_genfromtxt = np.genfromtxt

        def genfromtxt_with_rewrite(source, *args, **kwargs):
            table_file.write_text('99.0 99.0\n99.0 99.0\n')
            return orig_genfromtxt(source, *args, **kwargs)

        monkeypatch.setattr(np, 'genfromtxt', genfromtxt_with_rewrite)

        arr = read_cached_table(table_file)
        assert np.array_equal(arr, orig_data)

        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()
        with np.load(cache_file) as npz:
            cached_digest = str(npz['source_digest'])
            cached_data = np.array(npz['data'])

        assert cached_digest == orig_digest
        assert np.array_equal(cached_data, orig_data)

    def test_missing_source_raises_filenotfound(self, tmp_path: Path) -> None:
        """Verify non-existent table file raises FileNotFoundError."""
        missing = tmp_path / 'does_not_exist.dat'
        with pytest.raises(FileNotFoundError, match='Table file not found'):
            read_cached_table(missing)

    def test_cr_only_line_endings_match_base_genfromtxt(self, tmp_path: Path) -> None:
        """Verify CR-only line endings match base genfromtxt and cache correctly."""
        table_file = tmp_path / 'cr_table.dat'
        table_file.write_bytes(b'# header line\r1.0 2.0\r3.0 4.0\r5.0 6.0\r')

        expected = np.genfromtxt(table_file, skip_header=1)
        arr_cold = read_cached_table(table_file, skiprows=1)
        assert np.array_equal(arr_cold, expected)

        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert cache_file.is_file()

        arr_warm = read_cached_table(table_file, skiprows=1)
        assert np.array_equal(arr_warm, expected)

    def test_empty_parse_leaves_no_cache_file(self, tmp_path: Path) -> None:
        """Verify an empty or header-only file leaves no cache file."""
        table_file = tmp_path / 'empty_table.dat'
        table_file.write_text('# header only\n# second header\n')

        arr = read_cached_table(table_file, skiprows=2)
        assert arr.size == 0

        cache_file = table_file.with_name(f'{table_file.name}.cache.npz')
        assert not cache_file.exists()

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
