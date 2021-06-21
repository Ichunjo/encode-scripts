"""Automation module"""

__all__ = [
    'Parser', 'RunnerConfig', 'SelfRunner',
]

import argparse
from os import remove
from typing import NamedTuple, Optional, Sequence, Set, Tuple

import vapoursynth as vs

from .colors import Colors
from .config import FileInfo
from .tooling import (AudioCutter, AudioEncoder, BasicTool, LosslessEncoder,
                      Mux, VideoEncoder)
from .vpathlib import AnyPath

core = vs.core


class Parser():  # noqa
    def __init__(self, file: FileInfo) -> None:
        parser = argparse.ArgumentParser(description=f'Encode {file.name}')
        parser.add_argument('-L', '--lossless', action='store_true', default=False,
                            help='Write a lossless file instead of piping the pre-processing.')
        parser.add_argument('-Q', '--qpfile', action='store_true', default=False,
                            help='Write a qpfile from scene changes before encoding')
        parser.add_argument("-S", '--start', nargs='?', type=int, help='Start encode at frame START.')
        parser.add_argument("-E", '--end', nargs='?', type=int, help='Stop encode at frame END (inclusive).')
        self.args = parser.parse_args()
        super().__init__()

    def parsing(self, file: FileInfo, clip: vs.VideoNode) -> Tuple[FileInfo, vs.VideoNode]:  # noqa
        # Lossless check
        if self.args.lossless:
            file.do_lossless = True

        # Qpfile check
        if self.args.qpfile:
            file.do_qpfile = True

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


class RunnerConfig(NamedTuple):
    """Config for the SelfRunner"""
    v_encoder: VideoEncoder
    v_lossless_encoder: Optional[LosslessEncoder] = None
    a_extracters: Optional[Sequence[BasicTool]] = None
    a_cutters: Optional[Sequence[AudioCutter]] = None
    a_encoders: Optional[Sequence[AudioEncoder]] = None
    muxer: Optional[Mux] = None


class SelfRunner:
    """Self runner interface"""
    clip: vs.VideoNode
    file: FileInfo
    config: RunnerConfig

    cleanup: Set[AnyPath]

    def __init__(self, clip: vs.VideoNode, file: FileInfo, /, config: RunnerConfig) -> None:
        self.clip = clip
        self.file = file
        self.config = config
        self.cleanup = set()


    def run(self) -> None:
        """Tool chain"""
        self._parsing()
        self._encode()
        self._audio_getter()
        self._muxer()

    # @abstractmethod
    # def chapter(self) -> None:
    #     """Chapterisation"""
    #     # Examples
    #     assert self.file.chapter
    #     assert self.file.frame_start

    #     # Variables
    #     chap_names: List[Optional[str]] = []
    #     chapters: List[Chapter] = []
    #     fps: Fraction = self.clip.fps

    #     # XML or OGM chapters
    #     chap = MatroskaXMLChapters(self.file.chapter)  # type: ignore
    #     chap = OGMChapters(self.file.chapter)  # type: ignore

    #     # Method to be used
    #     chap.create(chapters, fps)
    #     chap.set_names(chap_names)
    #     chap.copy(Path(self.file.chapter).parent / 'new_chap.xml')
    #     chap.shift_times(0 - self.file.frame_start, fps)
    #     chap.create_qpfile(self.file.qpfile, fps)

    #     self.file.chapter = str(chap.chapter_file)

    def _parsing(self) -> None:
        parser = Parser(self.file)
        self.file, self.clip = parser.parsing(self.file, self.clip)

    def _encode(self) -> None:
        if self.file.do_lossless and self.config.v_lossless_encoder:
            if not self.file.name_clip_output_lossless.exists():
                self.config.v_lossless_encoder.run_enc(self.clip, self.file)
            self.clip = core.lsmas.LWLibavSource(self.file.name_clip_output_lossless.to_str())

        if not self.file.name_clip_output.exists():
            self.config.v_encoder.run_enc(self.clip, self.file)
            self.cleanup.add(self.file.name_clip_output)

    def _audio_getter(self) -> None:
        if self.config.a_extracters:
            for i, a_extracter in enumerate(self.config.a_extracters, start=1):
                if self.file.a_src and not self.file.a_src.format(i).exists():
                    a_extracter.run()
                    self.cleanup.add(self.file.a_src.format(i))

        if self.config.a_cutters:
            for i, a_cutter in enumerate(self.config.a_cutters, start=1):
                if self.file.a_src_cut and not self.file.a_src_cut.format(i).exists():
                    a_cutter.run()
                    self.cleanup.add(self.file.a_src_cut.format(i))

        if self.config.a_encoders:
            for i, a_encoder in enumerate(self.config.a_encoders, start=1):
                if self.file.a_enc_cut and not self.file.a_enc_cut.format(i).exists():
                    a_encoder.run()
                    self.cleanup.add(self.file.a_enc_cut.format(i))

    def _muxer(self) -> None:
        if self.config.muxer:
            wfs = self.config.muxer.run()
            self.cleanup.update(wfs)

    def do_cleanup(self, *extra_files: AnyPath) -> None:
        """Delete working files"""
        self.cleanup.update(extra_files)
        for files in self.cleanup:
            remove(files)
        self.cleanup.clear()
