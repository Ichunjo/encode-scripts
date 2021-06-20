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



BasicTools = Optional[Union[BasicTool, Sequence[BasicTool]]]
AudioCutters = Optional[Union[AudioCutter, Sequence[AudioCutter]]]
AudioEncoders = Optional[Union[AudioEncoder, Sequence[AudioEncoder]]]


class EncodeGoBrr:
    """Self runner interface"""
    clip: vs.VideoNode
    file: FileInfo
    v_encoder: VideoEncoder
    v_lossless_encoder: Optional[LosslessEncoder]
    a_extracters: List[BasicTool]
    a_cutters: List[AudioCutter]
    a_encoders: List[AudioEncoder]

    def __init__(self,
                 clip: vs.VideoNode, file: FileInfo, /,
                 v_encoder: VideoEncoder, v_lossless_encoder: Optional[LosslessEncoder] = None,
                 a_extracters: BasicTools = None,
                 a_cutters: AudioCutters = None,
                 a_encoders: AudioEncoders = None) -> None:
        """
        Args:
            clip (vs.VideoNode):
                (Filtered) clip.
            file (FileInfo):
                FileInfo object.

            v_encoder (VideoEncoder):
                Video encoder(s) used.

            v_lossless_encoder (Optional[LosslessEncoder]):
                Lossless encoder used if necessary.

            a_extracters (Union[BasicTool, Sequence[BasicTool]]):
                Audio extracter(s) used.

            a_cutters (Optional[Union[AudioCutter, Sequence[AudioCutter]]]):
                Audio cutter(s) used.

            a_encoders (Optional[Union[AudioEncoder, Sequence[AudioEncoder]]]):
                Audio encoder(s) used.
        """
        self.clip = clip
        self.file = file
        self.v_lossless_encoder = v_lossless_encoder
        self.v_encoder = v_encoder


        if a_extracters:
            self.a_extracters = list(a_extracters) if isinstance(a_extracters, Sequence) else [a_extracters]
        else:
            self.a_extracters = []

        if a_cutters:
            self.a_cutters = list(a_cutters) if isinstance(a_cutters, Sequence) else [a_cutters]
        else:
            self.a_cutters = []

        if a_encoders:
            self.a_encoders = list(a_encoders) if isinstance(a_encoders, Sequence) else [a_encoders]
        else:
            self.a_encoders = []


        super().__init__()


    def run(self) -> None:
        """Tool chain"""
        self._parsing()
        self._encode()
        self._audio_getter()

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

    # def merge(self) -> None:
    #     """Merge function"""
    #     video = VideoStream(
    #         path=Path(self.file.name_clip_output),
    #         name='HEVC BDRip by Vardë@Raws-Maji',
    #         lang=JAPANESE
    #     )

    #     assert self.file.a_enc_cut
    #     assert self.file.chapter
    #     BasicTool('mkvmerge', [
    #         '-o', self.file.name_file_final,
    #         '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', self.file.name_clip_output,
    #         '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', self.file.a_enc_cut.format(1),
    #         '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 5.1', '--language', '0:jpn', self.file.a_enc_cut.format(2),
    #         '--chapter-language', 'jpn', '--chapters', self.file.chapter
    #     ]).run()


    def _parsing(self) -> None:
        parser = Parser(self.file)
        self.file, self.clip = parser.parsing(self.file, self.clip)

    def _encode(self) -> None:
        if self.file.do_lossless and self.v_lossless_encoder:
            if not self.file.name_clip_output_lossless.exists():
                self.v_lossless_encoder.run_enc(self.clip, self.file)
            self.clip = core.lsmas.LWLibavSource(str(self.file.name_clip_output_lossless))

        if not self.file.name_clip_output.exists():
            self.v_encoder.run_enc(self.clip, self.file)

    def _audio_getter(self) -> None:
        for i, a_extracter in enumerate(self.a_extracters, start=1):
            assert self.file.a_src
            if not self.file.a_src.format(i).exists():
                a_extracter.run()

        for i, a_cutter in enumerate(self.a_cutters, start=1):
            assert self.file.a_src_cut
            if not self.file.a_src_cut.format(i).exists():
                a_cutter.run()

        for i, a_encoder in enumerate(self.a_encoders, start=1):
            assert self.file.a_enc_cut
            if not self.file.a_enc_cut.format(i).exists():
                a_encoder.run()

    def cleanup(self, **kwargs: Any) -> None:  # noqa
        self.file.cleanup(**kwargs)

    def do_cleanup(self, *extra_files: AnyPath) -> None:
        """Delete working files"""
        self.cleanup.update(extra_files)
        for files in self.cleanup:
            remove(files)
        self.cleanup.clear()
