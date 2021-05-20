# noqa
from __future__ import annotations

from pathlib import Path

from acsuite import eztrim
from vardautomation import BasicTool, FileInfo, X265Encoder

import vapoursynth as vs
core = vs.core



class EncodeGoBrrr():  # noqa
    def __init__(self: EncodeGoBrrr, clip: vs.VideoNode, file: FileInfo) -> None:
        self.clip = clip
        self.file = file

    def run(self: EncodeGoBrrr) -> None:  # noqa
        self._encode()
        self._audio_getter()
        self._merge()

    def _encode(self: EncodeGoBrrr) -> None:
        if not Path(self.file.name_clip_output):
            X265Encoder('x265', Path('churutto_common/x265_settings'), self.clip, self.file,
                        progress_update=lambda v, e:
                            print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end=""))

    def _audio_getter(self: EncodeGoBrrr) -> None:
        assert self.file.a_src.format(1)
        if not Path(self.file.a_src.format(1)).exists():
            BasicTool('mkvextract', [self.file.src, 'tracks', f'1:{self.file.a_src.format(1)}'])

        assert self.file.a_src_cut.format(1)
        if not Path(self.file.a_src_cut.format(1)).exists():
            eztrim(self.file.clip, (self.file.frame_start, self.file.frame_end), self.file.a_src.format(1), self.file.a_src_cut.format(1))

    def _merge(self: EncodeGoBrrr) -> None:
        assert self.file.a_src_cut.format(1)
        BasicTool('mkvmerge', [
            '-o', self.file.name_file_final,
            '--track-name', '0:HEVC WEBRip by Vardë@Owlolf', '--language', '0:jpn', self.file.name_clip_output,
            '--track-name', '0:EAC3 2.0', '--language', '0:jpn', self.file.a_src_cut.format(1),
        ])
