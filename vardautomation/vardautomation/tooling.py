"""Tooling module"""

__all__ = ['Tool', 'BasicTool',
           'AudioEncoder', 'QAACEncoder', 'FlacCompressionLevel', 'FlacEncoder',
           'AudioCutter',
           'VideoEncoder', 'X265Encoder', 'X264Encoder', 'LosslessEncoder',
           'progress_update_func',
           'Mux', 'Stream', 'MediaStream', 'VideoStream', 'AudioStream', 'ChapterStream',
           # Type:
           'AudioStreams']

import re
import subprocess
from abc import ABC, abstractmethod
from enum import IntEnum
from pprint import pformat
from typing import (Any, BinaryIO, Callable, Dict, List, Optional, Sequence,
                    Set, Tuple, Union, cast)

import vapoursynth as vs
from acsuite import eztrim
from lvsfunc.render import SceneChangeMode, find_scene_changes
from lxml import etree

from .colors import Colors
from .config import FileInfo
from .language import UNDEFINED, Lang
from .properties import Properties
from .vpathlib import AnyPath, VPath


class Tool(ABC):
    """Abstract Tool interface"""

    """Abstract tooling interface"""
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]]) -> None:
        self.binary = binary
        self.settings = settings
        self.params: List[str] = []
        super().__init__()

    @abstractmethod
    def run(self) -> None:
        """Tooling chain"""

    @abstractmethod
    def set_variable(self) -> Dict[str, Any]:
        """Set variables in the settings"""

    def _get_settings(self) -> None:
        if isinstance(self.settings, list):
            self.params = self.settings
        else:
            with open(self.settings, 'r') as sttgs:
                self.params = re.split(r'[\n\s]\s*', sttgs.read())

        self.params.insert(0, self.binary)

        self.params = [p.format(**self.set_variable()) for p in self.params]


class BasicTool(Tool):
    """BasicTool interface"""
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /, file: Optional[FileInfo] = None) -> None:
        """Helper allowing the use of CLI programs for basic tasks like video or audio track extraction.

        Args:
            binary (str):
                Path to your binary file.

            settings (Union[Path, List[str]]):
                Path to your settings file or list of string containing your settings.

            file (Optional[FileInfo]):
                FileInfo object. Not used in BasicTool implementation.
        """
        self.file = file
        super().__init__(binary, settings)

    def run(self) -> None:
        self._get_settings()
        self._do_tooling()

    def set_variable(self) -> Dict[str, Any]:
        return {}

    def _do_tooling(self) -> None:
        print(f'{Colors.INFO}{self.binary} command:', ' '.join(self.params) + f'{Colors.RESET}\n')
        subprocess.run(self.params, check=True, text=True, encoding='utf-8')


class AudioEncoder(BasicTool):
    """BasicTool interface for audio encoding"""
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /,
                 file: FileInfo, *, track: int, xml_tag: Optional[AnyPath] = None) -> None:
        super().__init__(binary, settings, file=file)
        self.track = track
        self.xml_tag = xml_tag

    def run(self) -> None:
        self._get_settings()
        self._do_tooling()
        if self.xml_tag:
            self._write_encoder_name_file()

    def set_variable(self) -> Dict[str, Any]:
        if self.file is None:
            raise ValueError('AudioEncoder: `file` is needed!')
        if self.file.a_src_cut is None:
            raise ValueError('AudioEncoder: `file.a_src_cut` is needed!')
        if self.file.a_enc_cut is None:
            raise ValueError('AudioEncoder: `file.a_enc_cut` is needed!')

        return dict(a_src_cut=self.file.a_src_cut.format(self.track).to_str(),
                    a_enc_cut=self.file.a_enc_cut.format(self.track).to_str())


    def _write_encoder_name_file(self) -> None:
        assert (a_enc_sut := self.file.a_enc_cut)

        tags = etree.Element('Tags')
        tag = etree.SubElement(tags, 'Tag')
        _ = etree.SubElement(tag, 'Targets')
        simple = etree.SubElement(tag, 'Simple')
        etree.SubElement(simple, 'Name').text = 'ENCODER'
        etree.SubElement(simple, 'String').text = Properties.get_encoder_name(a_enc_sut.format(self.track))

        assert self.xml_tag
        with open(self.file.workdir / self.xml_tag, 'wb') as f:
            f.write(
                etree.tostring(tags, encoding='utf-8', xml_declaration=True, pretty_print=True)
            )


class QAACEncoder(AudioEncoder):
    """QAAC AudioEncoder"""
    def __init__(self, /, file: FileInfo, *,
                 track: int, xml_tag: Optional[AnyPath] = None,
                 tvbr_quality: int = 127, qaac_args: Optional[List[str]] = None) -> None:
        binary = 'qaac'
        settings = ['{a_src_cut:s}', '-V', str(tvbr_quality), '--no-delay', '-o', '{a_enc_cut:s}']
        if qaac_args is not None:
            settings.append(*qaac_args)
        super().__init__(binary, settings, file, track=track, xml_tag=xml_tag)


class FlacCompressionLevel(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    FAST = 0
    BEST = 8
    VARDOU = 99


class FlacEncoder(AudioEncoder):
    """Flac AudioEncoder"""
    def __init__(self, file: FileInfo, *,
                 track: int, xml_tag: Optional[AnyPath] = None,
                 level: FlacCompressionLevel = FlacCompressionLevel.VARDOU,
                 use_ffmpeg: bool = True, flac_args: Optional[List[str]] = None) -> None:
        """
        Args:
            track (int):
                Track number

            level (FlacCompressionLevel, optional):
                See FlacCompressionLevel for all levels available.
                Defaults to FlacCompressionLevel.VARDOU.

            use_ffmpeg (bool, optional):
                Will use flac if false.
                Defaults to True.
        """
        if use_ffmpeg:
            binary = 'ffmpeg'
            if level == FlacCompressionLevel.VARDOU:
                level_args = ['-compression_level 12', '-lpc_type', 'cholesky',
                              '-lpc_passes', '3', '-exact_rice_parameters', '1']
            else:
                level_args = [f'-compression_level {level}']
            settings = ['-i', '{a_src_cut:s}', *level_args]
            if flac_args is not None:
                settings.append(*flac_args)
            settings += ['{a_enc_cut:s}']
        else:
            binary = 'flac'
            if level <= FlacCompressionLevel.EIGHT:
                if flac_args is not None:
                    settings = [*flac_args]
                else:
                    settings = []
                settings = [f'-{level}', '-o', '{a_enc_cut:s}', '{a_src_cut:s}']
            else:
                raise ValueError('FlacEncoder: "level" must be <= 8 if use_ffmpeg is false')
        super().__init__(binary, settings, file, track=track, xml_tag=xml_tag)


class AudioCutter():
    """Audio cutter using eztrim"""
    def __init__(self, file: FileInfo, /, *, track: int, **kwargs) -> None:
        self.file = file
        self.track = track
        self.kwargs = kwargs
        super().__init__()

    def run(self) -> None:  # noqa
        assert self.file.a_src
        assert self.file.a_src_cut
        eztrim(self.file.clip, (self.file.frame_start, self.file.frame_end),
               str(self.file.a_src.format(self.track)), str(self.file.a_src_cut.format(self.track)),
               **self.kwargs)




def progress_update_func(value: int, endvalue: int) -> None:
    """Callback function used in clip.output"""
    return print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end="")


class VideoEncoder(Tool):
    """VideoEncoder interface"""
    file: FileInfo
    clip: vs.VideoNode
    bits: int

    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = progress_update_func) -> None:
        """Helper intended to facilitate video encoding

        Args:
            binary (str):
                Path to your binary file.

            settings (Union[Path, List[str]]):
                Path to your settings file or list of string containing your settings.

            file (FileInfo):
                FileInfo object.

            progress_update (Optional[Callable[[int, int], None]], optional):
                Current progress can be reported by passing a callback function
                of the form func(current_frame, total_frames) to progress_update.
                Defaults to progress_update_func.
        """
        self.progress_update = progress_update
        super().__init__(binary, settings)

    def run_enc(self, clip: vs.VideoNode, file: FileInfo) -> None:
        """Run encoding"""
        self.file = file
        self.clip = clip

        assert self.clip.format
        self.bits = self.clip.format.bits_per_sample

        self._get_settings()

        if self.file.do_qpfile:
            self._create_qpfile()
            self.params += ['--qpfile', str(self.file.qpfile)]

        self._do_encode()

    def run(self) -> None:
        raise NameError('Use `run_enc` instead')

    def set_variable(self) -> Dict[str, Any]:
        return dict(clip_output=str(self.file.name_clip_output), filename=self.file.name)

    def _create_qpfile(self) -> None:
        if not (qpfile := self.file.qpfile).exists():
            scenes = find_scene_changes(self.clip, SceneChangeMode.WWXD_SCXVID_UNION)

            with qpfile.open('w') as qpf:
                qpf.writelines([f"{s} K" for s in scenes])

    def _do_encode(self) -> None:
        print(Colors.INFO)
        print('VideoEncoder command:', " ".join(self.params))
        print(f'{Colors.RESET}\n')

        with subprocess.Popen(self.params, stdin=subprocess.PIPE) as process:
            self.clip.output(cast(BinaryIO, process.stdin), y4m=True, progress_update=self.progress_update)


class X265Encoder(VideoEncoder):
    """Video encoder using x265 in HEVC"""
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = progress_update_func) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        min_luma, max_luma = Properties.get_color_range(self.params, self.clip, self.bits)
        return dict(clip_output=str(self.file.name_clip_output), filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits,
                    min_luma=min_luma, max_luma=max_luma)


class X264Encoder(VideoEncoder):
    """Video encoder using x264 in AVC"""
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = progress_update_func) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        csp = Properties.get_csp(self.clip)
        return dict(clip_output=str(self.file.name_clip_output), filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits, csp=csp)


class LosslessEncoder(VideoEncoder):  # noqa
    def __init__(self, binary: str, settings: Union[AnyPath, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = None) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        return dict(clip_output_lossless=str(self.file.name_clip_output_lossless))


class Stream:
    path: VPath

    def __init__(self, path: AnyPath) -> None:
        self.path = VPath(path)

    def __repr__(self) -> str:
        return pformat(vars(self), indent=1, width=80, sort_dicts=True)


class MediaStream(Stream):
    name: Optional[str] = None
    lang: Lang = UNDEFINED
    tag_file: Optional[VPath] = None

    def __init__(self, path: AnyPath, name: Optional[str] = None,
                 lang: Lang = UNDEFINED, tag_file: Optional[AnyPath] = None) -> None:
        super().__init__(path)
        self.name = name
        self.lang = lang
        self.tag_file = VPath(tag_file) if tag_file is not None else tag_file


class VideoStream(MediaStream):
    def __init__(self, path: AnyPath, name: Optional[str] = None,
                 lang: Lang = UNDEFINED, tag_file: Optional[AnyPath] = None) -> None:
        super().__init__(path, name=name, lang=lang, tag_file=tag_file)


class AudioStream(MediaStream):
    def __init__(self, path: AnyPath, name: Optional[str] = None,
                 lang: Lang = UNDEFINED, tag_file: Optional[AnyPath] = None) -> None:
        super().__init__(path, name=name, lang=lang, tag_file=tag_file)


class ChapterStream(Stream):
    lang: Lang = UNDEFINED
    charset: Optional[str] = None

    def __init__(self, path: AnyPath,
                 lang: Lang = UNDEFINED, charset: Optional[str] = None) -> None:
        super().__init__(path)
        self.lang = lang
        self.charset = charset



AudioStreams = Union[AudioStream, Sequence[AudioStream]]


class Mux:
    """Muxing interface using mkvmerge"""
    output: VPath
    video: VideoStream
    audios: List[AudioStream]
    chapters: Optional[ChapterStream]
    mkvmerge_path: VPath

    def __init__(self, file: FileInfo,
                 streams: Optional[Tuple[VideoStream, Optional[AudioStreams], Optional[ChapterStream]]] = None, *,
                 mkvmerge_path: AnyPath = VPath('mkvmerge')) -> None:
        """
            If `file` is specified:
                - Will find `file.name_file_final` as VideoStream
                - Will try to find in this order file.a_enc_cut, file.a_src_cut, file.a_src as long as there is a file.a_xxxx.format(n)
                - All languages are set to `und` and names to None.
            Otherwise will mux the `streams` to `output` if specified.
        """
        self.mkvmerge_path = VPath(mkvmerge_path)

        if file and not streams:
            self.output = file.name_file_final
            self.video = VideoStream(file.name_clip_output)

            self.audios = []

            i = 1
            while True:
                if (audio_path := file.a_enc_cut) and audio_path.format(i).exists():
                    self.audios += [AudioStream(audio_path)]
                elif (audio_path := file.a_src_cut) and audio_path.format(i).exists():
                    self.audios += [AudioStream(audio_path)]
                elif (audio_path := file.a_src) and audio_path.format(i).exists():
                    self.audios += [AudioStream(audio_path)]
                else:
                    break
                i += 1

            if file.chapter and (chap := file.chapter).exists():
                self.chapters = ChapterStream(chap)

        elif file and streams:
            self.output = file.name_file_final
            self.video, audios, self.chapters = streams
            if not audios:
                self.audios = []
            else:
                self.audios = [audios] if isinstance(audios, AudioStream) else list(audios)
        else:
            raise ValueError('Mux:')

    def run(self) -> Set[VPath]:
        """Make and launch the command"""
        cmd = ['-o', str(self.output)]

        work_files: Set[VPath] = set()

        if self.video.tag_file:
            cmd += ['--tags', '0:' + str(self.video.tag_file)]
        if self.video.name:
            cmd += ['--track-name', '0:' + self.video.name]
        cmd += ['--language', '0:' + self.video.lang.iso639, str(self.video.path)]
        work_files.add(self.video.path)

        if self.audios:
            for audio in self.audios:
                if audio.tag_file:
                    cmd += ['--tags', '0:' + str(audio.tag_file)]
                if audio.name:
                    cmd += ['--track-name', '0:' + audio.name]
                cmd += ['--language', '0:' + audio.lang.iso639, str(audio.path)]
                work_files.add(audio.path)

        if self.chapters:
            cmd += ['--chapter-language', self.chapters.lang.iso639]
            if self.chapters.charset:
                cmd += ['--chapter-charset', self.chapters.charset]
            cmd += ['--chapters', str(self.chapters.path)]
            work_files.add(self.chapters.path)

        BasicTool(str(self.mkvmerge_path), cmd).run()
        return work_files
