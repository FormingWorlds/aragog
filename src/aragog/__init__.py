"""Package level variables and initialises the package logger"""

from __future__ import annotations

__version__: str = '26.01.06'

import importlib.resources
import logging
import os
from importlib.resources.abc import Traversable

CFG_DATA: Traversable = importlib.resources.files(f'{__package__}.cfg')

# Creates the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = '[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()]'
    fmt += ' - %(message)s'
    datefmt: str = '%Y-%m-%d %H:%M:%S'
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = '[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s'
    datefmt: str = '%H:%M:%S'
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Sets up debug logging to the console.

    Returns:
        A logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger


def aragog_file_logger(
    console_level=logging.INFO, file_level=logging.DEBUG, log_dir=os.getcwd()
) -> logging.Logger:
    """Sets up console logging and file logging according to arguments.

    Returns:
        A logger
    """
    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(file_level)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)
    # File logger
    log_file = os.path.join(log_dir, f'{__package__}.log')
    file_handler: logging.Handler = logging.FileHandler(log_file)
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)

    return package_logger


# Expose public API. Imported below the logger setup to avoid circular
# imports: the solver module imports from aragog (this file) at module
# load time.
from aragog.solver import EntropySolver as EntropySolver  # noqa: E402
