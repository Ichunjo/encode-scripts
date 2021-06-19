"""pathlib.Path inheritance"""

__all__ = ['VPath']

from pathlib import Path


class VPath(Path):
    _flavour = type(Path())._flavour  # type: ignore

    def format(self, *args, **kwargs):
        """
            vpath.format(*args, **kwargs) -> VPath

            Return a formatted version of `vpath`, using substitutions from args and kwargs.
            The substitutions are identified by braces ('{' and '}')
        """
        return VPath(str(self).format(*args, **kwargs))
