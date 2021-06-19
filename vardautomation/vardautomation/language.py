"""Language module"""

__all__ = ['Language',
           'FRENCH', 'ENGLISH', 'JAPANESE', 'UNDEFINED']


from pprint import pformat

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

    @staticmethod
    def make(ietf: str):
        """Make a new Lang based on IETF"""
        return Lang(Language.make(ietf))



FRENCH = Lang(Language.make('fr'))
ENGLISH = Lang(Language.make('en'))
JAPANESE = Lang(Language.make('ja'))
UNDEFINED = Lang(Language.make())
