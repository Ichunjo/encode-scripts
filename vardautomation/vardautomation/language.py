"""Language module"""
from __future__ import annotations

__all__ = ['Language',
           'FRENCH', 'ENGLISH', 'JAPANESE', 'UNDEFINED']


from pprint import pformat
from typing import Optional

from langcodes import Language


class Lang:
    """Language"""
    name: str
    ietf: str
    iso639: str

    def __init__(self, language: Language, *, iso639_variant: str = 'B') -> None:
        self.name = language.autonym()
        self.ietf = str(language)
        self.iso639 = language.to_alpha3(variant=iso639_variant)

    def __repr__(self) -> str:
        return pformat(vars(self), indent=4, width=200, sort_dicts=False)

    @classmethod
    def make(cls, ietf: Optional[str]) -> Lang:
        """Make a new Lang based on IETF"""
        return cls(Language.make(ietf))


FRENCH = Lang.make('fr')
ENGLISH = Lang.make('en')
JAPANESE = Lang.make('ja')
UNDEFINED = Lang.make(None)
