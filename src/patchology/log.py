"""Log level per library from ``<PACKAGE>_LOG_LEVEL`` environment variables.

loguru has one handler for the whole process, so :func:`configure_logging`
installs a single stderr handler whose filter gives each message the level set
for its top-level package: ``PATCHOLOGY_LOG_LEVEL`` for ``patchology.*``,
``PATHOTRAIN_LOG_LEVEL`` for ``pathotrain.*``, and so on, ``INFO`` when unset.
``ALL`` means every level (``TRACE`` and up)::

    export PATCHOLOGY_LOG_LEVEL=WARNING
    export PATHOTRAIN_LOG_LEVEL=DEBUG
"""

import functools
import os
import sys

from loguru import logger

DEFAULT_LOG_LEVEL = "INFO"


@functools.cache
def level_for(package: str) -> int:
    """Minimum severity logged for ``package``, read from ``<PACKAGE>_LOG_LEVEL`` once.

    Raises:
        ValueError: The variable names no loguru level.
    """
    name = os.environ.get(f"{package.upper()}_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return logger.level("TRACE" if name == "ALL" else name).no


def _filter(record: dict) -> bool:
    return record["level"].no >= level_for((record["name"] or "").partition(".")[0])


def configure_logging(package: str) -> None:
    """Check ``package``'s level, then replace loguru's default handler (id 0) with the
    per-package level filter.

    The handler is left alone if the default one is already gone, i.e. the
    application or another library configured logging first, so every library
    can call this on import.

    Raises:
        ValueError: ``<PACKAGE>_LOG_LEVEL`` names no loguru level.
    """
    level_for(package)
    try:
        logger.remove(0)
    except ValueError:
        return
    logger.add(sys.stderr, filter=_filter, level=0)
