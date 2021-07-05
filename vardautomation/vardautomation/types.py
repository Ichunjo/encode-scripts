from os import PathLike
from typing import Callable, Optional, Tuple, Union

from lxml import etree
from vapoursynth import VideoNode

AnyPath = Union[PathLike[str], str]
Element = etree._Element
UpdateFunc = Callable[[int, int], None]
VPSIdx = Callable[[str], VideoNode]
Range = Union[Optional[int], Tuple[Optional[int], Optional[int]]]
