# noqa
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from acsuite import eztrim
from vardautomation import (BasicTool, FileInfo, NoPreset, Preset,
                            VideoEncoder, X265Encoder, Colors)

import vapoursynth as vs
core = vs.core


class FileInfoMore(FileInfo):  # noqa
    def __init__(self: FileInfoMore, src: str, /, frame_start: Optional[int], frame_end: Optional[int], *,
                 idx: Optional[Callable[[str], vs.VideoNode]] = None,
                 preset: Union[List[Preset], Preset] = NoPreset) -> None:
        super().__init__(src, frame_start=frame_start, frame_end=frame_end, idx=idx, preset=preset)

        self.name_clip_output_lossless = self.name + '_lossless.mkv'
        self.do_lossless = False

        self.a_src, self.a_src_cut, self.a_enc_cut = [
            s.format(1) for s in [self.a_src, self.a_src_cut, self.a_enc_cut]]

        self.chapter = f'_chapters/{self.name}.txt'


class NvencEncoder(VideoEncoder):  # noqa
    def __init__(self: NvencEncoder, binary: str, settings: Union[Path, List[str]],
                 clip: vs.VideoNode, file: FileInfoMore, /,
                 progress_update: Optional[Callable[[int, int], None]]) -> None:
        super().__init__(binary, settings, clip, file, progress_update=progress_update)
        self.file: FileInfoMore

    def set_variable(self: NvencEncoder) -> Dict[str, Any]:
        return dict(clip_output_lossless=self.file.name_clip_output_lossless)


class AudioEncoder(BasicTool):  # noqa
    def __init__(self: BasicTool, binary: str, settings: Union[Path, List[str]], /,
                 file: Optional[FileInfoMore]) -> None:
        super().__init__(binary, settings, file=file)

    def set_variable(self: BasicTool) -> Dict[str, Any]:
        return dict(a_src_cut=self.file.a_src_cut, a_enc_cut=self.file.a_enc_cut)


class Parser():  # noqa
    def __init__(self: Parser, file: FileInfoMore) -> None:
        parser = argparse.ArgumentParser(description=f'Encode {file.name}')
        parser.add_argument('-L', '--lossless', action='store_true', default=False,
                            help='Write a lossless file instead of piping the pre-processing.')
        parser.add_argument("-S", '--start', nargs='?', type=int, help='Start encode at frame START.')
        parser.add_argument("-E", '--end', nargs='?', type=int, help='Stop encode at frame END (inclusive).')
        self.args = parser.parse_args()


    def parsing(self: Parser, file: FileInfoMore, clip: vs.VideoNode) -> Tuple[FileInfoMore, vs.VideoNode]:  # noqa
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

        file.frame_start = file_frame_start
        file.frame_end = file_frame_end

        # print(frame_start, frame_end)
        # print(file.frame_start, file.frame_end)

        # # start frame check
        # if self.args.start is not None:
        #     frame_start = self.args.start
        #     # Set file.frame_start to 0 if it’s None and then add it
        #     # TODO: support negative frame_start
        #     if file.frame_start is None:
        #         file.frame_start = 0
        #     file.frame_start += self.args.start
        # else:
        #     frame_start = 0

        # # end frame check
        # if self.args.end is not None:
        #     # Set file.frame_end to num_frames if it’s None
        #     if file.frame_end is None:
        #         file.frame_end = file.clip.num_frames
        #     # If frame_end is negative
        #     if self.args.end < 0:
        #         frame_end = self.args.end
        #         file.frame_end += self.args.end
        #     # If frame_end is positive
        #     else:
        #         frame_end = self.args.end + 1
        #         # If file.frame_end is negative
        #         if file.frame_end < 0:
        #             file.frame_end = file.frame_start + self.args.end + 1 - self.args.start
        #         # if file.frame_end is postive
        #         else:
        #             file.frame_end = file.frame_start + self.args.end
        # # if frame_end is None
        # else:
        #     frame_end = file.clip_cut.num_frames

        return file, clip[frame_start:frame_end]


class EncodeGoBrrr():  # noqa
    def __init__(self: EncodeGoBrrr, clip: vs.VideoNode, file: FileInfoMore) -> None:
        self.clip = clip
        self.file = file

    def run(self: EncodeGoBrrr):  # noqa
        self._parsing()
        self._encode()
        self._audio_getter()
        self._merge()

    def _parsing(self: EncodeGoBrrr):
        parser = Parser(self.file)
        self.file, self.clip = parser.parsing(self.file, self.clip)

    def _encode(self: EncodeGoBrrr):
        if self.file.do_lossless:
            NvencEncoder('C:/NVEncC_5.30_x64/NVEncC64.exe', Path('love_live_common/nvenc_settings'),
                         self.clip, self.file, progress_update=None)
            self.clip = core.lsmas.LWLibavSource(self.file.name_clip_output_lossless)

        X265Encoder('x265', Path('love_live_common/x265_settings'), self.clip, self.file,
                    progress_update=lambda v, e:
                        print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end=""))

    def _audio_getter(self: EncodeGoBrrr):
        assert self.file.a_src
        if not Path(self.file.a_src).exists():
            BasicTool('eac3to', [self.file.src, '2:', self.file.a_src, '-log=NUL'])

        assert self.file.a_src_cut
        if not Path(self.file.a_src_cut).exists():
            eztrim(self.file.clip, (self.file.frame_start, self.file.frame_end), self.file.a_src, self.file.a_src_cut)

        assert self.file.a_enc_cut
        if not Path(self.file.a_enc_cut).exists():
            AudioEncoder('ffmpeg', Path('love_live_common/flac_settings'), self.file)

    def _merge(self: EncodeGoBrrr):
        assert self.file.a_enc_cut
        assert self.file.chapter
        BasicTool('mkvmerge', [
            '-o', self.file.name_file_final,
            '--track-name', '0:HEVC BDRip by Vardë@Manga-Family', '--language', '0:jpn', self.file.name_clip_output,
            '--track-name', '0:FLAC 2.0', '--language', '0:jpn', self.file.a_enc_cut,
            '--chapter-language', 'fre', '--chapters', self.file.chapter
        ])

    def cleanup(self: EncodeGoBrrr, **kwargs: Dict[str, bool]) -> None:  # noqa
        self.file.cleanup(**kwargs)
