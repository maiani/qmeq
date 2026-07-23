"""Backend selection and compiled-extension discovery.

Set ``QMEQ_BACKEND`` before importing :mod:`qmeq`:

``auto``
    Use the Cython extensions when the complete extension set is available,
    otherwise use the pure-Python implementations.
``python``
    Always use the pure-Python implementations.
``cython``
    Require the Cython extensions and fail if they cannot be imported.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from types import ModuleType

BACKEND_ENV = "QMEQ_BACKEND"
VALID_BACKENDS = ("auto", "python", "cython")


class BackendConfigurationError(ValueError):
    """Raised when the requested backend mode is invalid."""


class BackendUnavailableError(ImportError):
    """Raised when a requested or partially installed backend cannot load."""


def _read_requested_backend() -> str:
    value = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    if value not in VALID_BACKENDS:
        choices = ", ".join(VALID_BACKENDS)
        raise BackendConfigurationError(
            f"Invalid {BACKEND_ENV}={value!r}; expected one of: {choices}."
        )
    return value


_REQUESTED_BACKEND = _read_requested_backend()
_GROUPS: dict[str, str] = {}
_FALLBACK_REASONS: dict[str, str] = {}


def get_requested_backend() -> str:
    """Return the backend mode requested before QmeQ was imported."""

    return _REQUESTED_BACKEND


def get_backend() -> str:
    """Return the backend currently active across registered component groups."""

    active = set(_GROUPS.values())
    if not active:
        return "uninitialized"
    if len(active) == 1:
        return active.pop()
    return "mixed"


def get_backend_status() -> dict[str, object]:
    """Return JSON-serializable backend information for diagnostics."""

    return {
        "requested": get_requested_backend(),
        "active": get_backend(),
        "groups": dict(sorted(_GROUPS.items())),
        "fallback_reasons": dict(sorted(_FALLBACK_REASONS.items())),
    }


def _unavailable_message(group: str, module_name: str) -> str:
    return (
        f"The Cython backend is unavailable for {group!r}: could not import "
        f"{module_name!r}. Rebuild the extensions or set "
        f"{BACKEND_ENV}=python before importing qmeq."
    )


def load_compiled_modules(
    group: str, module_names: Iterable[str]
) -> dict[str, ModuleType] | None:
    """Load an atomic group of extensions according to the selected backend.

    ``None`` means that callers should use their pure-Python implementations.
    A missing extension permits fallback only in ``auto`` mode and only when no
    Cython component has loaded. Import failures from a present extension, or a
    partial extension installation, are always reported as errors.
    """

    names = tuple(module_names)
    if _REQUESTED_BACKEND == "python":
        _GROUPS[group] = "python"
        return None

    loaded: dict[str, ModuleType] = {}
    for module_name in names:
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            target_is_missing = exc.name == module_name
            cython_already_active = "cython" in _GROUPS.values()
            partial_group = bool(loaded)
            if (
                _REQUESTED_BACKEND == "cython"
                or not target_is_missing
                or cython_already_active
                or partial_group
            ):
                raise BackendUnavailableError(
                    _unavailable_message(group, module_name)
                ) from exc
            _GROUPS[group] = "python"
            _FALLBACK_REASONS[group] = f"{module_name} is not installed"
            return None
        except ImportError as exc:
            raise BackendUnavailableError(
                _unavailable_message(group, module_name)
            ) from exc

    _GROUPS[group] = "cython"
    return loaded
