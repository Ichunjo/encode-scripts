"""Automation module"""
import argparse
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Any, BinaryIO, Callable, Dict, List, Optional, Tuple,
                    Union, cast)

import vapoursynth as vs

from .clip_settings import ClipSettings
from .colors import Colors
from .presets import NoPreset, Preset

core = vs.core

# TODO: Better english because I’m fucking bad


class FileInfo():  # noqa: PLR0902
    """File info object"""
    path: str
    src: str
    idx: Optional[Callable[[str], vs.VideoNode]]
    preset: List[Preset]

    name: str

    a_src: Optional[str]
    a_src_cut: Optional[str]
    a_enc_cut: Optional[str]
    chapter: Optional[str]

    clip: vs.VideoNode
    frame_start: Optional[int]
    frame_end: Optional[int]
    clip_cut: vs.VideoNode

    name_clip_output: str
    name_file_final: str

    name_clip_output_lossless: str
    do_lossless: bool


    def __init__(self, src: str, /,
                 frame_start: Optional[int] = None, frame_end: Optional[int] = None, *,
                 idx: Optional[Callable[[str], vs.VideoNode]] = None, preset: Union[List[Preset], Preset] = NoPreset) -> None:
        """Helper which allows to store the data related to your file to be encoded

        Args:
            src (str):
                Path to your source file.

            frame_start (Optional[int], optional):
                Number of frames to trim at the beginning of the clip. Python slicing.
                Defaults to None.

            frame_end (Optional[int], optional):
                Number of frames to trim at the end of the clip. Python slicing.
                Defaults to None.

            idx (Optional[Callable[[str], vs.VideoNode]], optional):
                Indexer used to index the video track.
                Defaults to lvsfunc.misc.source.

            preset (Union[List[Preset], Preset], optional):
                Preset used to fill idx, a_src, a_src_cut, a_enc_cut and chapter attributes.
                Defaults to NoPreset.
        """
        self.path = str(Path(src).parent.joinpath(Path(src).stem))
        self.src = src
        self.idx = idx

        self.name = Path(sys.argv[0]).stem

        self.a_src = None
        self.a_src_cut = None
        self.a_enc_cut = None
        self.chapter = None

        self.preset = [preset] if isinstance(preset, Preset) else preset
        self._params_fill_preset()

        self.clip = self.idx(src)
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.clip_cut = self.clip[self.frame_start:self.frame_end] if (self.frame_start or self.frame_end) else self.clip

        self.name_clip_output = self.name + '.265'
        self.name_file_final = self.name + '.mkv'

        self.name_clip_output_lossless = self.name + '_lossless.mkv'
        self.do_lossless = False

        super().__init__()

    def __str__(self) -> str:
        txt = 'File infos:\n'
        txt += f'Full path: {self.src}\n'
        txt += f'Clip format: \n{self.clip_cut.format}\n'
        txt += f'Name: {self.name}\n'
        return txt

    def _params_fill_preset(self) -> None:
        for pre in self.preset:
            for d1, d2 in zip(self.__dict__.items(), pre.__dict__.items()):  # noqa: PLC0103
                _, v = d1  # noqa: PLC0103
                kp, vp = d2  # noqa: PLC0103
                if isinstance(vp, str):
                    vp = vp.format(path=self.path, name=self.name, num='{}')  # noqa: PLC0103
                setattr(self, kp, vp if not v else v)

    def cleanup(self, *,  # noqa
                a_src: bool = True, a_src_cut: bool = True, a_enc_cut: bool = True,
                chapter: bool = False, name_clip_output: bool = False) -> None:
        files = (self.a_src, self.a_src_cut, self.a_enc_cut, self.chapter, self.name_clip_output)
        booleans = (a_src, a_src_cut, a_enc_cut, chapter, name_clip_output)

        for file, boolean in zip(files, booleans):
            if boolean and file:
                os.remove(file)

    def set_audio_track_to_1(self) -> None:  # noqa
        self.a_src, self.a_src_cut, self.a_enc_cut = [
            s.format(1) for s in [self.a_src, self.a_src_cut, self.a_enc_cut]]



class Tool(ABC):
    """Abstract Encoder interface"""
    def __init__(self, binary: str, settings: Union[Path, List[str]]) -> None:
        self.binary = binary
        self.settings = settings
        self.params: List[str] = []
        super().__init__()

    @abstractmethod
    def run(self) -> None:
        """Tooling chain"""
        pass  # noqa: PLC0103, PLW0107

    @abstractmethod
    def set_variable(self) -> Dict[str, Any]:
        """Set variables in the settings"""
        pass  # noqa: PLC0103, PLW0107

    def _get_settings(self) -> None:
        if isinstance(self.settings, Path):
            with open(self.settings, 'r') as sttgs:
                self.params = re.split(r'[\n\s]\s*', sttgs.read())
        elif isinstance(self.settings, list):
            self.params = self.settings
        else:
            print(Colors.FAIL)
            raise ValueError('Tool: "settings" parameter must be a Path or a list of string')

        self.params.insert(0, self.binary)

        self.params = [p.format(**self.set_variable()) for p in self.params]


class BasicTool(Tool):
    """BasicTool interface"""
    def __init__(self, binary: str, settings: Union[Path, List[str]], /, file: Optional[FileInfo] = None) -> None:
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
        self.run()

    def run(self) -> None:
        self._get_settings()
        self._do_tooling()

    def set_variable(self) -> Dict[str, Any]:
        return {}

    def _do_tooling(self) -> None:
        print(Colors.INFO)
        print(f'{self.binary} command:', ' '.join(self.params))
        print(f'{Colors.RESET}\n')
        subprocess.run(self.params, check=True, text=True, encoding='utf-8')


class AudioEncoder(BasicTool):  # noqa
    def __init__(self, binary: str, settings: Union[Path, List[str]], /,
                 file: Optional[FileInfo], *, track: Optional[int]) -> None:
        super().__init__(binary, settings, file=file)
        self.track = track

    def set_variable(self) -> Dict[str, Any]:
        if self.track:
            dico = dict(a_src_cut=self.file.a_src_cut.format(self.track),
                        a_enc_cut=self.file.a_enc_cut.format(self.track))
        else:
            dico = dict(a_src_cut=self.file.a_src_cut,
                        a_enc_cut=self.file.a_enc_cut)
        return dico


class VideoEncoder(Tool):
    """VideoEncoder interface"""
    def __init__(self, binary: str, settings: Union[Path, List[str]], clip: vs.VideoNode,
                 file: FileInfo, /, progress_update: Optional[Callable[[int, int], None]] = None) -> None:
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
                Defaults to None.
        """
        self.file = file
        self.progress_update = progress_update
        self.clip = clip
        self.bits = self.clip.format.bits_per_sample
        super().__init__(binary, settings)
        self.run()

    def run(self) -> None:
        self._get_settings()
        self._do_encode()

    def set_variable(self) -> Dict[str, Any]:
        return dict(clip_output=self.file.name_clip_output, filename=self.file.name)

    def _do_encode(self) -> None:
        print(Colors.INFO)
        print('VideoEncoder command:', " ".join(self.params))
        print(f'{Colors.RESET}\n')

        with subprocess.Popen(self.params, stdin=subprocess.PIPE) as process:
            self.clip.output(cast(BinaryIO, process.stdin), y4m=True, progress_update=self.progress_update)


class X265Encoder(VideoEncoder):
    """Video encoder using x265 in HEVC"""
    def __init__(self, binary: str, settings: Union[Path, List[str]], clip: vs.VideoNode,
                 file: FileInfo, /, progress_update: Optional[Callable[[int, int], None]]) -> None:
        super().__init__(binary, settings, clip, file, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        min_luma, max_luma = ClipSettings.get_color_range(self.params, self.clip, self.bits)
        return dict(clip_output=self.file.name_clip_output, filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits,
                    min_luma=min_luma, max_luma=max_luma)


class X264Encoder(VideoEncoder):
    """Video encoder using x264 in AVC"""
    def __init__(self, binary: str, settings: Union[Path, List[str]], clip: vs.VideoNode,
                 file: FileInfo, /, progress_update: Optional[Callable[[int, int], None]]) -> None:
        super().__init__(binary, settings, clip, file, progress_update=progress_update)

    def run(self) -> None:
        self._get_settings()
        self._do_encode()

    def set_variable(self) -> Dict[str, Any]:
        csp = ClipSettings.get_csp(self.clip)
        return dict(clip_output=self.file.name_clip_output, filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits,
                    csp=csp)


class LosslessEncoder(VideoEncoder):  # noqa
    def __init__(self, binary: str, settings: Union[Path, List[str]],
                 clip: vs.VideoNode, file: FileInfo, /,
                 progress_update: Optional[Callable[[int, int], None]]) -> None:
        super().__init__(binary, settings, clip, file, progress_update=progress_update)
        self.file: FileInfo

    def set_variable(self) -> Dict[str, Any]:
        return dict(clip_output_lossless=self.file.name_clip_output_lossless)



class Parser():  # noqa
    def __init__(self, file: FileInfo) -> None:
        parser = argparse.ArgumentParser(description=f'Encode {file.name}')
        parser.add_argument('-L', '--lossless', action='store_true', default=False,
                            help='Write a lossless file instead of piping the pre-processing.')
        parser.add_argument("-S", '--start', nargs='?', type=int, help='Start encode at frame START.')
        parser.add_argument("-E", '--end', nargs='?', type=int, help='Stop encode at frame END (inclusive).')
        self.args = parser.parse_args()


    def parsing(self, file: FileInfo, clip: vs.VideoNode) -> Tuple[FileInfo, vs.VideoNode]:  # noqa
        # Lossless check
        if self.args.lossless:
            file.do_lossless = True

        file_frame_start: Optional[int] = None
        file_frame_end: Optional[int] = None

        frame_start: Optional[int] = None
        frame_end: Optional[int] = None

        # start frame check
        if self.args.start is not None:
            if self.args.start >= 0:
                frame_start = self.args.start
                if file.frame_start is None:
                    file.frame_start = 0
                file_frame_start = file.frame_start + self.args.start
            else:
                print(Colors.FAIL)
                raise ValueError('--start START must be a positive value!')
        else:
            file_frame_start = file.frame_start

        # end frame check
        if self.args.end is not None:
            if self.args.end >= 0:
                frame_end = self.args.end + 1
                if file.frame_end is None:
                    file.frame_end = file.clip.num_frames
                file_frame_end = min(file.frame_start + self.args.end + 1,
                                     file.frame_end)
            else:
                print(Colors.FAIL)
                raise ValueError('--end END must be a positive value!')
        else:
            file_frame_end = file.frame_end

        file.frame_start = file_frame_start
        file.frame_end = file_frame_end

        return file, clip[frame_start:frame_end]
