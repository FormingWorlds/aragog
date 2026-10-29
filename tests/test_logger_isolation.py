"""The aragog package logger starts every test with its import level and handlers.

``aragog run`` configures the ``fwl.aragog`` logger for the rest of the process (DEBUG level, a
console and a file handler). A test that calls it through ``CliRunner`` must not pass that state to
the tests that run after it in the same process.
"""

from __future__ import annotations

import logging

import pytest

import aragog
from tests.conftest import package_logger_isolation

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_aragog_file_logger_configures_the_package_logger(tmp_path):
    """The setup the fixture undoes: DEBUG level, a console and a file handler, no NullHandler."""
    aragog.aragog_file_logger(log_dir=str(tmp_path))
    assert aragog.logger.level == logging.DEBUG
    assert sorted(type(h).__name__ for h in aragog.logger.handlers) == [
        'FileHandler',
        'StreamHandler',
    ]


def test_the_package_logger_starts_at_its_import_state():
    """The package logger has no level and only the NullHandler that ``aragog`` adds at import."""
    assert aragog.logger.level == logging.NOTSET
    assert [type(h) for h in aragog.logger.handlers] == [logging.NullHandler]


def test_leaving_the_block_closes_every_file_handler_a_cli_run_added(tmp_path):
    """A second ``aragog_file_logger`` call detaches the first file handler; both are closed."""
    import_handler = aragog.logger.handlers[0]
    with package_logger_isolation():
        aragog.aragog_file_logger(log_dir=str(tmp_path))
        first = [h for h in aragog.logger.handlers if isinstance(h, logging.FileHandler)]
        aragog.aragog_file_logger(log_dir=str(tmp_path))
        second = [h for h in aragog.logger.handlers if isinstance(h, logging.FileHandler)]
        assert aragog.logger.level == logging.DEBUG
        assert first[0] not in aragog.logger.handlers
        assert import_handler not in aragog.logger.handlers
    assert first[0].stream is None
    assert second[0].stream is None
    assert aragog.logger.handlers == [import_handler]
    assert aragog.logger.level == logging.NOTSET


def test_leaving_the_block_keeps_a_handler_that_was_there_before(tmp_path):
    """A handler attached before the block stays open, one appended in it is closed; flags come back."""
    before = logging.FileHandler(tmp_path / 'before.log')
    aragog.logger.addHandler(before)
    aragog.logger.setLevel(logging.INFO)
    aragog.logger.propagate = False
    aragog.logger.disabled = True
    with package_logger_isolation():
        aragog.logger.addHandler(logging.FileHandler(tmp_path / 'during.log'))
        appended = logging.FileHandler(tmp_path / 'appended.log')
        aragog.logger.handlers.append(appended)
        aragog.logger.setLevel(logging.DEBUG)
        aragog.logger.propagate = True
        aragog.logger.disabled = False
    assert before in aragog.logger.handlers
    assert before.stream is not None
    assert appended.stream is None
    assert len(aragog.logger.handlers) == 2
    assert aragog.logger.level == logging.INFO
    assert aragog.logger.propagate is False
    assert aragog.logger.disabled is True


def test_leaving_the_block_clears_the_cached_level_of_the_logger():
    """A DEBUG check cached in the block does not outlive it; the logger follows its parent again."""
    with package_logger_isolation():
        aragog.logger.setLevel(logging.DEBUG)
        assert aragog.logger.isEnabledFor(logging.DEBUG)
    parent = aragog.logger.parent
    assert aragog.logger.isEnabledFor(logging.DEBUG) == parent.isEnabledFor(logging.DEBUG)
    assert aragog.logger.getEffectiveLevel() == parent.getEffectiveLevel()


def test_an_inner_block_keeps_the_outer_block_recording(tmp_path):
    """After a nested block ends, a handler added and detached later is still closed by the outer one."""
    fixture_recorder = vars(aragog.logger).get('addHandler')
    with package_logger_isolation():
        outer_recorder = vars(aragog.logger)['addHandler']
        with package_logger_isolation():
            pass
        assert vars(aragog.logger)['addHandler'] is outer_recorder
        aragog.aragog_file_logger(log_dir=str(tmp_path))
        first = [h for h in aragog.logger.handlers if isinstance(h, logging.FileHandler)]
        aragog.aragog_file_logger(log_dir=str(tmp_path))
        assert first[0] not in aragog.logger.handlers
    assert first[0].stream is None
    assert [type(h) for h in aragog.logger.handlers] == [logging.NullHandler]
    assert vars(aragog.logger).get('addHandler') is fixture_recorder


def test_a_block_that_raises_still_restores_the_logger(tmp_path):
    """An exception in the block does not skip the restore or the close of the added handlers."""
    import_handler = aragog.logger.handlers[0]
    fixture_recorder = vars(aragog.logger).get('addHandler')
    with pytest.raises(RuntimeError, match='raised in the block'):
        with package_logger_isolation():
            aragog.aragog_file_logger(log_dir=str(tmp_path))
            added = [h for h in aragog.logger.handlers if isinstance(h, logging.FileHandler)]
            raise RuntimeError('raised in the block')
    assert added[0].stream is None
    assert aragog.logger.handlers == [import_handler]
    assert aragog.logger.level == logging.NOTSET
    assert vars(aragog.logger).get('addHandler') is fixture_recorder
