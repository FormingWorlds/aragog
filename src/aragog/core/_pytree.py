"""JAX pytree registration of the core classes, so that compiled programs take them as arguments.

Floats, NumPy and JAX arrays and registered objects (and tuples, lists and dicts of them) are
leaves, floats as strong float64; strings, booleans, integers and None are static and compared by
value; any other object (a SciPy table) is static and compared by identity. Cached properties
and callables (compiled functions an object keeps) are left out and built again when used.
"""

from __future__ import annotations

import functools

import jax
import numpy as np

_REGISTERED: set[type] = set()
_PLAIN = (str, bool, int, type(None))


class _Same:
    """A static value compared by identity: equal only to a holder of the same object."""

    __slots__ = ('value',)

    def __init__(self, value) -> None:
        self.value = value

    def __eq__(self, other) -> bool:
        return isinstance(other, _Same) and other.value is self.value

    def __hash__(self) -> int:
        return id(self.value)


def _is_data(v) -> bool:
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (float, np.floating, np.ndarray, jax.Array)) or type(v) in _REGISTERED:
        return True
    if isinstance(v, (tuple, list)):
        return len(v) > 0 and all(_is_data(x) for x in v)
    if isinstance(v, dict):
        return len(v) > 0 and all(_is_data(x) for x in v.values())
    return False


def _leaf(v):
    if isinstance(v, (float, np.floating)):
        return np.float64(v)
    if isinstance(v, (tuple, list)):
        return type(v)(_leaf(x) for x in v)
    if isinstance(v, dict):
        return {k: _leaf(x) for k, x in v.items()}
    return v


def register_pytree(cls):
    """Register ``cls`` as a pytree over its instance attributes (see the module docstring)."""
    cached = frozenset(
        k
        for c in cls.__mro__
        for k, v in vars(c).items()
        if isinstance(v, functools.cached_property)
    )

    def flatten(obj):
        items = sorted(
            (k, v) for k, v in vars(obj).items() if k not in cached and not callable(v)
        )
        keys = tuple(k for k, v in items if _is_data(v))
        static = tuple(
            (k, v if isinstance(v, _PLAIN) else _Same(v)) for k, v in items if not _is_data(v)
        )
        return [_leaf(v) for k, v in items if _is_data(v)], (keys, static)

    def unflatten(aux, children):
        keys, static = aux
        obj = object.__new__(cls)
        obj.__dict__.update((k, v.value if isinstance(v, _Same) else v) for k, v in static)
        obj.__dict__.update(zip(keys, children))
        return obj

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _REGISTERED.add(cls)
    return cls
