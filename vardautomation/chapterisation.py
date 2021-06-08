"""Chapterisation module"""

__all__ = ['Language', 'Chapter', 'Chapters', 'OGMChapters', 'MatroskaXMLChapters',
           'create_qpfile',
           'FRENCH', 'ENGLISH', 'JAPANESE', 'UNDEFINED']

import os
import random
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import List, NamedTuple, Optional, Set, cast

from langcodes import Language as L
from lxml import etree
from prettyprinter import doc, pretty_call, pretty_repr, register_pretty
from prettyprinter.prettyprinter import PrettyContext

from .colors import Colors


class Language:
    """Language"""
    name: str
    ietf: str
    iso639: str

    def __init__(self, lang: L) -> None:
        self.name = lang.autonym()
        self.ietf = str(lang)
        self.iso639 = lang.to_alpha3(variant='B')

    def __repr__(self) -> str:
        @register_pretty(Language)
        def _repr(value: object, ctx: PrettyContext) -> doc.Doc:
            dic = vars(value)
            return pretty_call(ctx, Language, dic)

        return pretty_repr(self)


FRENCH = Language(L.make('fr'))
ENGLISH = Language(L.make('en'))
JAPANESE = Language(L.make('ja'))
UNDEFINED = Language(L.make())



class Chapter(NamedTuple):
    """Chapter object"""
    name: str
    start_frame: int
    end_frame: Optional[int] = None
    lang: Language = UNDEFINED


class Chapters(ABC):
    """Abtract chapters interface"""
    chapter_file: str

    def __init__(self, chapter_file: str) -> None:
        """Chapter file path as parameter"""
        self.chapter_file = chapter_file
        super().__init__()

    def __repr__(self) -> str:
        @register_pretty(Chapters)
        def _repr(value: object, ctx: PrettyContext) -> doc.Doc:
            dic = vars(value)
            return pretty_call(ctx, Chapters, dic)

        return pretty_repr(self)

    @abstractmethod
    def create(self, chapters: List[Chapter], fps: Fraction) -> None:
        """Create a chapter"""

    @abstractmethod
    def set_names(self, names: List[Optional[str]]) -> None:
        """Change chapter names."""

    def copy(self, destination: str) -> None:
        """Copy source chapter to destination."""
        os.system(f'copy "{self.chapter_file}" "{destination}"')

    def _logging(self, action: str) -> None:
        print(f'{Colors.INFO}Chapter file sucessfuly {action} at: {self.chapter_file}{Colors.RESET}\n')

    def _f2seconds(self, f: int, fps: Fraction, /) -> float:  # noqa
        if f == 0:
            s = 0.0  # noqa

        t = round(float(10 ** 9 * f * fps ** -1))  # noqa
        s = t / 10 ** 9  # noqa
        return s

    def _f2ts(self, f: int, fps: Fraction, /, *, precision: int = 3) -> str:  # noqa
        s = self._f2seconds(f, fps)  # noqa
        ts = self._seconds2ts(s, precision=precision)  # noqa
        return ts

    def _ts2seconds(self, ts: str, /) -> float:  # noqa
        h, m, s = map(float, ts.split(':'))  # noqa
        return h * 3600 + m * 60 + s

    def _seconds2ts(self, s: float, /, *, precision: int = 3) -> str:  # noqa
        m = s // 60  # noqa
        s %= 60  # noqa
        h = m // 60  # noqa
        m %= 60  # noqa

        return self._compose_ts(h, m, s, precision=precision)

    def _seconds2f(self, s: float, fps: Fraction, /) -> int:  # noqa
        return round(s * fps)

    def _ts2f(self, ts: str, fps: Fraction, /) -> int:
        s = self._ts2seconds(ts)  # noqa
        f = self._seconds2f(s, fps)  # noqa
        return f

    @staticmethod
    def _compose_ts(h: float, m: float, s: float, /, *, precision: int = 3) -> str:
        if precision == 0:  # noqa
            return f"{h:02.0f}:{m:02.0f}:{round(s):02}"
        elif precision == 3:
            return f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
        elif precision == 6:
            return f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
        elif precision == 9:
            return f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
        else:
            raise ValueError('precision must be <= 9 and >= 0')


class OGMChapters(Chapters):
    """OGMChapters object"""

    def create(self, chapters: List[Chapter], fps: Fraction) -> None:
        """Create a txt chapter file."""

        with open(self.chapter_file, 'w') as file:
            for i, chapter in enumerate(chapters):
                file.writelines([f'CHAPTER{i:02.0f}={self._f2ts(chapter.start_frame, fps)}\n',
                                 f'CHAPTER{i:02.0f}NAME={chapter.name}\n'])

        self._logging('created')

    def set_names(self, names: List[Optional[str]]) -> None:
        data = self._get_data()

        times = data[::2]
        old = data[1::2]

        if len(names) > len(old):
            raise ValueError('set_names: too many names!')
        if len(names) < len(old):
            names += [None] * (len(old) - len(names))

        new = [f'CHAPTER{i+1:02.0f}NAME={names[i]}\n' if names[i] is not None else chapname
               for i, chapname in enumerate(old)]

        with open(self.chapter_file, 'w') as file:
            file.writelines([val for tup in zip(times, new) for val in tup])

        self._logging('updated')

    def shift_times(self, frames: int, fps: Fraction) -> None:
        """Shift times by given number of frames."""
        data = self._get_data()

        shifttime = self._f2seconds(frames, fps)

        chaptimes = data[::2]
        chapnames = data[1::2]

        newchaptimes = [
            f'{chaptime.split("=")[0]}={self._seconds2ts(max(0, self._ts2seconds(chaptime.split("=")[1]) + shifttime))}\n'
            for chaptime in chaptimes
        ]

        with open(self.chapter_file, 'w') as file:
            file.writelines([val for tup in zip(newchaptimes, chapnames) for val in tup])

        self._logging('shifted')

    def ogm_to_chapters(self, fps: Fraction, lang: Language = UNDEFINED) -> List[Chapter]:
        """Convert OGM Chapters to a list of Chapter"""
        data = self._get_data()

        chaptimes = data[::2]
        chapnames = data[1::2]

        chapters = [
            Chapter(chapname.split('=')[1], self._ts2f(chaptime.split('=')[1], fps), lang=lang)
            for chaptime, chapname in zip(chaptimes, chapnames)
        ]

        return chapters

    def _get_data(self) -> List[str]:
        with open(self.chapter_file, 'r') as file:
            data = file.readlines()
        return data


class MatroskaXMLChapters(Chapters):
    """MatroskaXMLChapters object """
    fps: Fraction
    timecodes: List[float]

    ed_entry = 'EditionEntry'
    ed_uid = 'EditionUID'

    chap_atom = 'ChapterAtom'
    chap_start = 'ChapterTimeStart'
    chap_end = 'ChapterTimeEnd'
    chap_uid = 'ChapterUID'
    chap_disp = 'ChapterDisplay'
    chap_name = 'ChapterString'
    chap_ietf = 'ChapLanguageIETF'
    chap_iso639 = 'ChapterLanguage'

    doctype = '<!DOCTYPE Tags SYSTEM "matroskatags.dtd">'

    def create(self, chapters: List[Chapter], fps: Fraction) -> None:
        """Create a xml chapter file."""
        self.fps = fps

        root = etree.Element('Chapters')

        edit_entry = etree.SubElement(root, self.ed_entry)
        etree.SubElement(edit_entry, self.ed_uid).text = str(random.getrandbits(64))

        # Append chapters
        for chap in [self._make_chapter_xml(c) for c in chapters]:
            root.append(chap)

        with open(self.chapter_file, 'wb') as file:
            file.write(etree.tostring(
                root, 'utf-8', xml_declaration=True, pretty_print=True, doctype=self.doctype)
            )

        self._logging('created')

    def set_names(self, names: List[Optional[str]]) -> None:
        tree = self._get_tree()

        olds = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_name}')
        olds = cast(List[etree._Element], olds)  # noqa: PLW0212

        if len(names) > len(olds):
            raise ValueError('set_names: too many names!')
        if len(names) < len(olds):
            names += [None] * (len(olds) - len(names))

        for new, old in zip(names, olds):
            old.text = new

        with open(self.chapter_file, 'wb') as file:
            tree.write(file, pretty_print=True, xml_declaration=True, with_comments=True)

        self._logging('updated')

    def shift_times(self, frames: int, fps: Fraction) -> None:
        """Shift times by given number of frames."""
        shifttime = self._f2seconds(frames, fps)


        tree = self._get_tree()

        timestarts = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_start}')
        timestarts = cast(List[etree._Element], timestarts)  # noqa: PLW0212

        timeends = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_end}')
        timeends = cast(List[etree._Element], timeends)  # noqa: PLW0212

        for t_s in timestarts:
            if isinstance(t_s.text, str):
                t_s.text = self._seconds2ts(max(0, self._ts2seconds(t_s.text) + shifttime), precision=9)

        for t_e in timeends:
            if isinstance(t_e.text, str) and t_e.text != '':
                t_e.text = self._seconds2ts(max(0, self._ts2seconds(t_e.text) + shifttime), precision=9)


        with open(self.chapter_file, 'wb') as file:
            tree.write(file, pretty_print=True, xml_declaration=True, with_comments=True)

        self._logging('shifted')

    def xml_to_chapters(self, fps: Fraction, lang: Optional[Language] = None) -> List[Chapter]:
        """Convert XML Chapters to a list of Chapter"""
        tree = self._get_tree()

        timestarts = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_start}')
        timestarts = cast(List[etree._Element], timestarts)  # noqa: PLW0212

        timeends = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_end}')
        timeends = cast(List[etree._Element], timeends)  # noqa: PLW0212

        names = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_name}')
        names = cast(List[etree._Element], names)  # noqa: PLW0212

        ietfs = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_ietf}')
        ietfs = cast(List[etree._Element], ietfs)  # noqa: PLW0212

        iso639s = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_iso639}')
        iso639s = cast(List[etree._Element], iso639s)  # noqa: PLW0212

        if all(len(lst) != len(timestarts) for lst in {timeends, ietfs, iso639s}):
            raise ValueError('I don’t know how to fix that lmao')

        chapters = [
            Chapter(name=name.text if isinstance(name.text, str) else '',
                    start_frame=self._ts2f(timestart.text if isinstance(timestart.text, str) else '', fps),
                    end_frame=self._ts2f(timeend.text if isinstance(timeend.text, str) else '', fps),
                    lang=lang if lang is not None else Language('', ietf.text if isinstance(ietf.text, str) else '',
                                                                iso639.text if isinstance(iso639.text, str) else ''))
            for name, timestart, timeend, ietf, iso639 in zip(names, timestarts, timeends, ietfs, iso639s)
        ]

        return chapters


    def _make_chapter_xml(self, chapter: Chapter) -> etree._Element:  # noqa: PLW0212

        atom = etree.Element(self.chap_atom)


        etree.SubElement(atom, self.chap_start).text = self._f2ts(chapter.start_frame, self.fps, precision=9)
        if chapter.end_frame:
            etree.SubElement(atom, self.chap_end).text = self._f2ts(chapter.end_frame, self.fps, precision=9)

        etree.SubElement(atom, self.chap_uid).text = str(random.getrandbits(64))


        disp = etree.SubElement(atom, self.chap_disp)
        etree.SubElement(disp, self.chap_name).text = chapter.name
        etree.SubElement(disp, self.chap_ietf).text = chapter.lang.ietf
        etree.SubElement(disp, self.chap_iso639).text = chapter.lang.iso639


        return atom


    def _get_tree(self) -> etree._ElementTree:  # noqa: PLW0212
        return etree.parse(self.chapter_file)


def create_qpfile(qpfile: str,
                  frames: Optional[Set[int]] = None, *,
                  chapters: Optional[List[Chapter]] = None) -> None:
    """Create a qp file from a list of Chapter"""
    keyf: Set[int] = set()
    if chapters:
        for chap in chapters:
            keyf.add(chap.start_frame)
    elif frames:
        keyf = frames

    with open(qpfile, "w", encoding='utf-8') as qp:  # noqa: PLC0103
        qp.writelines([f"{f} K\n" for f in sorted(keyf)])
