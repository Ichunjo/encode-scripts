from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Union, cast

from lxml import etree
from vapoursynth import VideoNode

AnyPath = Union[PathLike[str], str]
Element = etree._Element
UpdateFunc = Callable[[int, int], None]
VPSIdx = Callable[[str], VideoNode]


class ElementTree(etree._ElementTree):  # type: ignore
    def xpath(self, _path: Union[str, bytes],  # type: ignore
              namespaces: Optional[Union[Dict[str, str], Dict[bytes, bytes]]] = None,  # type: ignore
              extensions: Any = None, smart_strings: bool = True,
              **_variables) -> List[Element]:  # type: ignore
        xpathobject = super().xpath(  # type: ignore
            _path, namespaces=namespaces, extensions=extensions,
            smart_strings=smart_strings, **_variables
        )
        return cast(List[Element], xpathobject)  # type: ignore
