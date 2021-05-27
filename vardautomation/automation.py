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


class Tool(ABC):
    """Abstract tooling interface"""
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
                 file: Optional[FileInfo], *, track: int) -> None:
        super().__init__(binary, settings, file=file)
        self.track = track

    def set_variable(self) -> Dict[str, Any]:
        assert self.file
        assert self.file.a_src_cut
        assert self.file.a_enc_cut
        dico = dict(a_src_cut=self.file.a_src_cut.format(self.track),
                    a_enc_cut=self.file.a_enc_cut.format(self.track))
        return dico


class AudioCutter():
    """Audio cutter using eztrim"""
    def __init__(self, file: Optional[FileInfo], /, *, track: int) -> None:
        self.file = file
        self.track = track
        super().__init__()

    def run(self) -> None:  # noqa
        assert self.file
        assert self.file.a_src
        assert self.file.a_src_cut
        eztrim(self.file.clip, (self.file.frame_start, self.file.frame_end),
               self.file.a_src.format(self.track), self.file.a_src_cut.format(self.track))


class VideoEncoder(Tool):
    """VideoEncoder interface"""
    file: FileInfo
    clip: vs.VideoNode
    bits: int

    def __init__(self, binary: str, settings: Union[Path, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = None) -> None:
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
        self.progress_update = progress_update
        super().__init__(binary, settings)

    def run_enc(self, clip: vs.VideoNode, file: FileInfo) -> None:
        """Run encoding"""
        self.file = file
        self.clip = clip
        assert self.clip.format
        self.bits = self.clip.format.bits_per_sample
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
    def __init__(self, binary: str, settings: Union[Path, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = None) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        assert self.clip
        min_luma, max_luma = Properties.get_color_range(self.params, self.clip, self.bits)
        return dict(clip_output=self.file.name_clip_output, filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits,
                    min_luma=min_luma, max_luma=max_luma)


class X264Encoder(VideoEncoder):
    """Video encoder using x264 in AVC"""
    def __init__(self, binary: str, settings: Union[Path, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = None) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

    def set_variable(self) -> Dict[str, Any]:
        assert self.clip
        csp = Properties.get_csp(self.clip)
        return dict(clip_output=self.file.name_clip_output, filename=self.file.name, frames=self.clip.num_frames,
                    fps_num=self.clip.fps.numerator, fps_den=self.clip.fps.denominator,
                    bits=self.bits, csp=csp)


class LosslessEncoder(VideoEncoder):  # noqa
    def __init__(self, binary: str, settings: Union[Path, List[str]], /,
                 progress_update: Optional[Callable[[int, int], None]] = None) -> None:
        super().__init__(binary, settings, progress_update=progress_update)

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
