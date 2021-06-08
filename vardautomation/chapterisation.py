"""Chapterisation module"""

__all__ = ['Language', 'Chapter', 'Chapters', 'OGMChapters', 'MatroskaXMLChapters',
           'create_qpfile',
           'FRENCH', 'ENGLISH', 'JAPANESE', 'UNDEFINED']

import os
import random
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path
from typing import List, NamedTuple, NoReturn, Optional, Sequence, Set, cast

from langcodes import Language as L
from lxml import etree
from prettyprinter import doc, pretty_call, pretty_repr, register_pretty
from prettyprinter.prettyprinter import PrettyContext
from pyparsebluray import mpls

from .colors import Colors
from .timeconv import Convert


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
        print(f'{Colors.INFO}Chapter file sucessfully {action} at: {self.chapter_file}{Colors.RESET}\n')


class OGMChapters(Chapters):
    """OGMChapters object"""

    def create(self, chapters: List[Chapter], fps: Fraction) -> None:
        """Create a txt chapter file."""

        with open(self.chapter_file, 'w') as file:
            for i, chapter in enumerate(chapters):
                file.writelines([f'CHAPTER{i:02.0f}={Convert.f2ts(chapter.start_frame, fps)}\n',
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

        shifttime = Convert.f2seconds(frames, fps)

        chaptimes = data[::2]
        chapnames = data[1::2]

        newchaptimes = [
            f'{chaptime.split("=")[0]}={Convert.seconds2ts(max(0, Convert.ts2seconds(chaptime.split("=")[1]) + shifttime))}\n'
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
            Chapter(chapname.split('=')[1], Convert.ts2f(chaptime.split('=')[1], fps), lang=lang)
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

    doctype = '<!-- <!DOCTYPE Tags SYSTEM "matroskatags.dtd"> -->'

    def create(self, chapters: List[Chapter], fps: Fraction) -> None:
        """Create a xml chapter file."""
        self.fps = fps

        root = etree.Element('Chapters')

        edit_entry = etree.SubElement(root, self.ed_entry)
        etree.SubElement(edit_entry, self.ed_uid).text = str(random.getrandbits(64))

        # Append chapters
        for chap in [self._make_chapter_xml(c) for c in chapters]:
            edit_entry.append(chap)

        with open(self.chapter_file, 'wb') as file:
            file.write(etree.tostring(
                root, encoding='utf-8', xml_declaration=True,
                pretty_print=True, doctype=self.doctype)
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
        tree = self._get_tree()

        shifttime = Convert.f2seconds(frames, fps)


        timestarts = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_start}')
        timestarts = cast(List[etree._Element], timestarts)  # noqa: PLW0212

        timeends = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_end}')
        timeends = cast(List[etree._Element], timeends)  # noqa: PLW0212

        for t_s in timestarts:
            if isinstance(t_s.text, str):
                t_s.text = Convert.seconds2ts(max(0, Convert.ts2seconds(t_s.text) + shifttime), precision=9)

        for t_e in timeends:
            if isinstance(t_e.text, str) and t_e.text != '':
                t_e.text = Convert.seconds2ts(max(0, Convert.ts2seconds(t_e.text) + shifttime), precision=9)


        with open(self.chapter_file, 'wb') as file:
            tree.write(file, pretty_print=True, xml_declaration=True, with_comments=True)

        self._logging('shifted')

    def xml_to_chapters(self, fps: Fraction, lang: Optional[Language] = None) -> List[Chapter]:
        """Convert XML Chapters to a list of Chapter"""
        tree = self._get_tree()

        timestarts = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_start}')
        timestarts = cast(List[etree._Element], timestarts)  # noqa: PLW0212


        timeends = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_end}')
        timeends = cast(List[Optional[etree._Element]], timeends)  # noqa: PLW0212
        if len(timeends) != len(timestarts):
            timeends += [None] * (len(timestarts) - len(timeends))


        names = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_name}')
        names = cast(List[Optional[etree._Element]], names)  # noqa: PLW0212
        if len(names) != len(timestarts):
            names += [None] * (len(timestarts) - len(names))


        ietfs = tree.xpath(f'/Chapters/{self.ed_entry}/{self.chap_atom}/{self.chap_disp}/{self.chap_ietf}')
        ietfs = cast(List[Optional[etree._Element]], ietfs)  # noqa: PLW0212
        if len(ietfs) != len(timestarts):
            ietfs += [None] * (len(timestarts) - len(ietfs))


        chapters: List[Chapter] = []
        for name, timestart, timeend, ietf in zip(names, timestarts, timeends, ietfs):

            name = name.text if isinstance(name.text, str) else ''

            if isinstance(timestart.text, str):
                start_frame = Convert.ts2f(timestart.text, fps)
            else:
                raise ValueError()

            try:
                end_frame = Convert.ts2f(timeend.text, fps)  # type: ignore
            except AttributeError:
                end_frame = None

            if not lang and isinstance(ietf.text, str):
                lang = Language(L.make(ietf.text))
            else:
                assert lang

            chapter = Chapter(name=name, start_frame=start_frame, end_frame=end_frame, lang=lang)
            chapters.append(chapter)

        return chapters


    def _make_chapter_xml(self, chapter: Chapter) -> etree._Element:  # noqa: PLW0212

        atom = etree.Element(self.chap_atom)


        etree.SubElement(atom, self.chap_start).text = Convert.f2ts(chapter.start_frame, self.fps, precision=9)
        if chapter.end_frame:
            etree.SubElement(atom, self.chap_end).text = Convert.f2ts(chapter.end_frame, self.fps, precision=9)

        etree.SubElement(atom, self.chap_uid).text = str(random.getrandbits(64))


        disp = etree.SubElement(atom, self.chap_disp)
        etree.SubElement(disp, self.chap_name).text = chapter.name
        etree.SubElement(disp, self.chap_ietf).text = chapter.lang.ietf
        etree.SubElement(disp, self.chap_iso639).text = chapter.lang.iso639


        return atom


    def _get_tree(self) -> etree._ElementTree:  # noqa: PLW0212
        return etree.parse(self.chapter_file)


def create_qpfile(qpfile: str,
                  frames: Optional[Sequence[int]] = None, *,
                  chapters: Optional[List[Chapter]] = None) -> None:
    """Create a qp file from a list of Chapter or frames"""
    keyf: Set[int] = set()
    if chapters:
        for chap in chapters:
            keyf.add(chap.start_frame)
    elif frames:
        keyf = set(frames)

    with open(qpfile, "w", encoding='utf-8') as qp:  # noqa: PLC0103
        qp.writelines([f"{f} K\n" for f in sorted(keyf)])



class MplsChapters(Chapters):
    """MplsChapters object"""
    m2ts: Path
    chapters: List[Chapter]
    fps: Fraction

    def create(self, chapters: List[Chapter], fps: Fraction) -> NoReturn:
        raise NotImplementedError("Can't create a mpls file!")

    def set_names(self, names: List[Optional[str]]) -> NoReturn:
        raise NotImplementedError("Can't change name from a mpls file!")


class MplsReader():
    """Mpls reader"""
    bd_folder: Path

    mpls_folder: Path
    m2ts_folder: Path

    set_lang: Language
    default_chap_name: str

    class MplsFile(NamedTuple):  # noqa: PLC0115
        mpls_file: Path
        mpls_chapters: List[MplsChapters]

    def __init__(self, bd_folder: Path = Path(), set_lang: Language = UNDEFINED, default_chap_name: str = 'Chapter') -> None:
        """Initialise a MplsReader.
           All parameters are optionnal if you just want to use the `parse_mpls` method.

        Args:
            bd_folder (Path, optional):
                A valid bluray folder path should contain a BDMV and CERTIFICATE folders.
                Defaults to pathlib.Path().

            set_lang (Language, optional):
                Language to be set. Defaults to UNDEFINED.

            default_chap_name (str, optional):
                Prefix used as default name for the generated chapters.
                Defaults to 'Chapter'.
        """
        self.bd_folder = bd_folder

        self.mpls_folder = self.bd_folder.joinpath('BDMV/PLAYLIST')
        self.m2ts_folder = self.bd_folder.joinpath('BDMV/STREAM')

        self.set_lang = set_lang
        self.default_chap_name = default_chap_name

    def get_playlist(self) -> List[MplsFile]:
        """Returns a list of all the mpls files contained in the folder specified in the constructor."""
        mpls_files = sorted(self.mpls_folder.glob('*.mpls'))

        return [
            self.MplsFile(mpls_file=mpls_file,
                          mpls_chapters=self.parse_mpls(mpls_file))
            for mpls_file in mpls_files
        ]

    def write_playlist(self, output_folder: Optional[Path] = None) -> None:
        """Extract and write the playlist folder to XML chapters files.

        Args:
            output_folder (Optional[Path], optional):
                Will write in the mpls folder if not specified.
                Defaults to None.
        """
        playlist = self.get_playlist()

        if not output_folder:
            output_folder = self.mpls_folder

        for mpls_file in playlist:
            for mpls_chapters in mpls_file.mpls_chapters:
                # Some mpls_chapters don't necessarily have attributes mpls_chapters.chapters
                try:
                    MatroskaXMLChapters(str(output_folder.joinpath(f'{mpls_file.mpls_file.stem}_{mpls_chapters.m2ts.stem}.xml'))) \
                        .create(mpls_chapters.chapters, mpls_chapters.fps)
                except AttributeError:
                    pass


    def parse_mpls(self, mpls_file: Path) -> List[MplsChapters]:
        """Parse a mpls file and return a list of chapters that were in the mpls file."""
        with open(mpls_file, 'rb') as file:
            header = mpls.load_movie_playlist(file)

            file.seek(header.playlist_start_address, os.SEEK_SET)
            playlist = mpls.load_playlist(file)

            if not playlist.play_items:
                raise ValueError('There is no play items in this file!')

            file.seek(header.playlist_mark_start_address, os.SEEK_SET)
            playlist_mark = mpls.load_playlist_mark(file)
            if (plsmarks := playlist_mark.playlist_marks) is not None:
                marks = plsmarks
            else:
                raise ValueError('There is no playlist marks in this file!')


            mpls_chaps: List[MplsChapters] = []

            for i, playitem in enumerate(playlist.play_items):

                # Create a MplsChapters and add its linked mpls
                mpls_chap = MplsChapters(str(mpls_file))

                # Add the m2ts name
                if (name := playitem.clip_information_filename) and \
                   (ext := playitem.clip_codec_identifier):
                    mpls_chap.m2ts = self.m2ts_folder.joinpath(f'{name}.{ext}'.lower())

                # Sort the chapters/marks linked to the current item
                linked_marks = [mark for mark in marks if mark.ref_to_play_item_id == i]

                # linked_marks could be empty
                if linked_marks:
                    # Extract the offset
                    assert playitem.intime
                    offset = min(playitem.intime, linked_marks[0].mark_timestamp)

                    # Extract the fps and store it
                    try:
                        assert (fps_n := playitem.stn_table.prim_video_stream_entries[0][1].framerate)
                        mpls_chap.fps = mpls.FRAMERATE[fps_n]
                    except KeyError as kerr:
                        raise ValueError('Framerate not found') from kerr

                    # Finally extract the chapters
                    mpls_chap.chapters = sorted(self._mplschapters_to_chapters(linked_marks, offset, mpls_chap.fps))

                # And add to the list
                mpls_chaps.append(mpls_chap)

            return mpls_chaps


    def _mplschapters_to_chapters(self, marks: List[mpls.playlist_mark.PlaylistMark], offset: int, fps: Fraction) -> Set[Chapter]:
        chapters: Set[Chapter] = set()
        for i, mark in enumerate(marks, start=1):
            chapters.add(
                Chapter(name=f'{self.default_chap_name} {i:02.0f}',
                        start_frame=Convert.seconds2f((mark.mark_timestamp - offset) / 45000, fps),
                        end_frame=None,
                        lang=self.set_lang))
        return chapters
