"""Slide readers with an OpenSlide-compatible interface.

Use :func:`open_slide` to get a :class:`SlideReader`; it picks the requested
backend and falls back to the other one if the slide cannot be opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from patchology.reader.base import SlideOpenError, SlideReader

__all__ = ["Backend", "SlideOpenError", "SlideReader", "open_slide"]


Backend = Literal["openslide", "fastslide"]
BACKENDS: tuple[Backend, ...] = ("openslide", "fastslide")


def open_slide(
    path: str | Path, backend: Backend = "openslide", *, fallback: bool = True
) -> SlideReader:
    """Open a whole slide image.

    Args:
        path: Slide file. For MIRAX this is the ``.mrxs`` file.
        backend: Preferred backend, ``"openslide"`` or ``"fastslide"``.
        fallback: If the preferred backend is not installed or fails to open
            the file, with any error, try the other backend before giving up.
            Errors while reading an opened slide are never caught, so one
            slide's tiles always come from one decoder.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        SlideOpenError: No backend could open the slide; it lists each
            backend's error, and the last unexpected one is its ``__cause__``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    order = [backend]
    if fallback:
        order += [b for b in BACKENDS if b != backend]

    errors = []
    cause = None
    for name in order:
        try:
            reader = _reader_class(name)(path)
        except (ImportError, SlideOpenError) as e:
            errors.append(f"{name}: {e}")
            continue
        except Exception as e:  # noqa: BLE001 - any open failure means try the other backend
            errors.append(f"{name}: {type(e).__name__}: {e}")
            cause = e
            continue
        if name != backend:
            logger.warning(
                "{} could not open {}, fell back to {} ({})", backend, path, name, errors[0]
            )
        logger.debug("opened {} with {}", path, name)
        return reader
    raise SlideOpenError(f"could not open {path}:\n  " + "\n  ".join(errors)) from cause


def _reader_class(name: Backend) -> type[SlideReader]:
    # Imported lazily so that a missing optional backend only matters when used.
    if name == "openslide":
        from patchology.reader.openslide_backend import OpenSlideReader

        return OpenSlideReader
    from patchology.reader.fastslide_backend import FastSlideReader

    return FastSlideReader
