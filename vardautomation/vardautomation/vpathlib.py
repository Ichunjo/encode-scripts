"""pathlib.Path inheritance"""

__all__ = ['VPath', 'AnyPath']

from os import PathLike
from pathlib import Path
from typing import Union


AnyPath = Union[PathLike, str]


class VPath(Path):
    _flavour = type(Path())._flavour  # type: ignore

    def format(self, *args, **kwargs):
        """
            vpath.format(*args, **kwargs) -> VPath

            Return a formatted version of `vpath`, using substitutions from args and kwargs.
            The substitutions are identified by braces ('{' and '}')
        """
        return VPath(str(self).format(*args, **kwargs))
