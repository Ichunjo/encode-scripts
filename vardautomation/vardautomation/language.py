"""Language module"""

__all__ = ['Language',
           'FRENCH', 'ENGLISH', 'JAPANESE', 'UNDEFINED']


from langcodes import Language
from prettyprinter import doc, pretty_call, pretty_repr, register_pretty
from prettyprinter.prettyprinter import PrettyContext


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
        @register_pretty(Lang)
        def _repr(value: object, ctx: PrettyContext) -> doc.Doc:
            dic = vars(value)
            return pretty_call(ctx, Lang, dic)

        return pretty_repr(self)

    @staticmethod
    def make(ietf: str):
        """Make a new Lang based on IETF"""
        return Lang(Language.make(ietf))



FRENCH = Lang(Language.make('fr'))
ENGLISH = Lang(Language.make('en'))
JAPANESE = Lang(Language.make('ja'))
UNDEFINED = Lang(Language.make())
