from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Union, cast

from lxml import etree
from vapoursynth import VideoNode

AnyPath = Union[PathLike[str], str]
Element = etree._Element
UpdateFunc = Callable[[int, int], None]
VPSIdx = Callable[[str], VideoNode]


class ElementTree(etree._ElementTree):
    def xpath(self, _path: Union[str, bytes],  # type: ignore
              namespaces: Optional[Union[Dict[str, str], Dict[bytes, bytes]]] = None,
              extensions: Any = None, smart_strings: bool = True,
              **_variables) -> List[Element]:
        xpathobject = super().xpath(
            _path, namespaces=namespaces, extensions=extensions,
            smart_strings=smart_strings, **_variables
        )
        return cast(List[Element], xpathobject)
