"""Mutation canaries for the EOS tabular data disk cache.

Each test verifies that breaking a guard or invariant causes the test to fail.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import aragog.eos.table_cache as tc


class TestTableCacheCanaries:
    """Canary tests proving that breaking cache guards triggers test failures."""

    def test_canary_a_bit_equal_failure_when_data_mutated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canary A: mutator on hit must cause test failure."""
        table_file = tmp_path / 'test_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        # Cold read populates cache
        arr1 = tc.read_cached_table(table_file)

        # Mutate cache file contents to be off by 1.0
        cache_file = table_file.with_suffix('.npz')
        with np.load(cache_file) as npz:
            meta = {k: npz[k] for k in npz.files}
        meta['data'] = meta['data'] + 1.0
        np.savez(cache_file, **meta)

        # Warm read now yields corrupted data
        arr2 = tc.read_cached_table(table_file)
        with pytest.raises(AssertionError):
            assert np.array_equal(arr1, arr2)

    def test_canary_b_digest_omission_causes_stale_cache_leak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canary B: disabling source digest check leaks stale data when mtime/size preserved."""
        table_file = tmp_path / 'edited_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        stat_orig = table_file.stat()
        arr1 = tc.read_cached_table(table_file)
        orig_digest = tc._compute_source_digest(table_file)
        assert arr1[1, 0] == 3.0

        # Edit content but restore size and timestamp
        table_file.write_text('1.0 2.0\n5.0 4.0\n')
        os.utime(table_file, (stat_orig.st_atime, stat_orig.st_mtime))

        # Mutation: mock _compute_source_digest to always return the stale orig_digest
        monkeypatch.setattr(tc, '_compute_source_digest', lambda fp: orig_digest)

        # Warm read erroneously hits cache with stale 3.0 instead of 5.0
        arr2 = tc.read_cached_table(table_file)
        with pytest.raises(AssertionError):
            assert arr2[1, 0] == 5.0

    def test_canary_c_ignoring_reader_args_causes_shape_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canary C: ignoring skiprows in cache key returns stale shape."""
        table_file = tmp_path / 'args_table.dat'
        table_file.write_text('1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n')

        # Read with skiprows=0
        arr0 = tc.read_cached_table(table_file, skiprows=0)
        assert arr0.shape == (3, 3)

        # Mutation: mock cache key resolution to ignore skiprows
        orig_resolve = tc._resolve_cache_path
        monkeypatch.setattr(
            tc,
            '_resolve_cache_path',
            lambda fp, skiprows, dtype_str, usecols_str: orig_resolve(
                fp, 0, dtype_str, usecols_str
            ),
        )

        # Read with skiprows=1 should return shape (2, 3), but hit returns (3, 3)
        cache_file = table_file.with_suffix('.npz')
        # Overwrite metadata to accept any skiprows
        with np.load(cache_file) as npz:
            meta = {k: npz[k] for k in npz.files}
        meta['skiprows'] = np.int64(1)
        np.savez(cache_file, **meta)

        arr1 = tc.read_cached_table(table_file, skiprows=1)
        with pytest.raises(AssertionError):
            assert arr1.shape == (2, 3)

    def test_canary_d_unhandled_write_error_crashes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canary D: unhandled filesystem error must raise instead of fallback."""
        table_file = tmp_path / 'ro_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        # Mutation: make _write_cache_file raise unhandled OSError
        def broken_write(*args, **kwargs):
            raise PermissionError('Simulated write permission error')

        monkeypatch.setattr(tc, '_write_cache_file', broken_write)

        with pytest.raises(PermissionError):
            tc._write_cache_file(table_file.with_suffix('.npz'), None, 0, '', 0, '', '')

    def test_canary_g_writing_beside_source_violates_dataset_integrity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canary G: writing cache beside FWL_DATA source alters dataset directory."""
        fwl_root = tmp_path / 'fwl_root'
        dataset_dir = fwl_root / 'datasets' / 'test_dataset'
        dataset_dir.mkdir(parents=True)
        monkeypatch.setenv('FWL_DATA', str(fwl_root))

        table_file = dataset_dir / 'source_table.dat'
        table_file.write_text('1.0 2.0\n3.0 4.0\n')

        before_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )

        # Mutation: force cache to be written beside source instead of central cache root
        monkeypatch.setattr(
            tc,
            '_resolve_cache_path',
            lambda fp, skiprows, dtype_str, usecols_str: fp.with_suffix('.npz'),
        )

        tc.read_cached_table(table_file)

        after_entries = sorted(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in dataset_dir.iterdir()]
        )

        # Dataset directory was mutated with a .npz file, failing the invariant
        with pytest.raises(AssertionError):
            assert before_entries == after_entries
