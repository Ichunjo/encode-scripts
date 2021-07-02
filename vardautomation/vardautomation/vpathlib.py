"""pathlib.Path inheritance"""

from __future__ import annotations

__all__ = ['VPath']

from pathlib import Path


class VPath(Path):
    """Vardë Path"""
    _flavour = type(Path())._flavour  # type: ignore

    def __format__(self, format_spec: str) -> str:
        return str(self)

    def format(self, *args, **kwargs) -> VPath:
        """
            vpath.format(*args, **kwargs) -> VPath

            Return a formatted version of `vpath`, using substitutions from args and kwargs.
            The substitutions are identified by braces ('{' and '}')
        """
        return VPath(self.to_str().format(*args, **kwargs))

    def to_str(self) -> str:
        """
            Return the string representation of the path, suitable for
            passing to system calls.
        """
        return str(self)
