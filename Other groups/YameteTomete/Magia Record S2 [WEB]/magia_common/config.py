from typing import List, Union

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, BasicTool, EztrimCutter,
                            FileInfo, Mux, Patch, RunnerConfig, SelfRunner,
                            VideoStream, X265Encoder)
from vardefunc.types import Range

core = vs.core


class Encoding:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip

    def run(self) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut

        self.v_encoder = X265Encoder('magia_common/x265_settings')
        a_extracters = BasicTool('mkvextract', [self.file.path.to_str(), 'tracks', f'1:{self.file.a_src.set_track(1)}'])
        a_cutters = EztrimCutter(self.file, track=1)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC WEBRip by Vardë@Raws-Maji', JAPANESE),
                AudioStream(self.file.a_src_cut.set_track(1), 'EAC3 2.0', JAPANESE),
                None
            ), merge_args={'--quiet': None, '--ui-language': 'en'}
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            a_extracters, a_cutters, None,
            muxer
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()
