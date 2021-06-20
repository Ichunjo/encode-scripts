"""pathlib.Path inheritance"""

__all__ = ['VPath', 'AnyPath']

from os import PathLike
from pathlib import Path
from typing import Union


AnyPath = Union[PathLike, str]


class VPath(Path):
    _flavour = type(Path())._flavour  # type: ignore

    def __format__(self, format_spec: str) -> str:
        return str(self)

    def format(self, *args, **kwargs):
        """
            vpath.format(*args, **kwargs) -> VPath

            Return a formatted version of `vpath`, using substitutions from args and kwargs.
            The substitutions are identified by braces ('{' and '}')
        """
        return VPath(str(self).format(*args, **kwargs))

    def to_str(self) -> str:
        """
            Return the string representation of the path, suitable for
            passing to system calls.
        """
        return str(self)
