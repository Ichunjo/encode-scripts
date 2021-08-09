from typing import List, Union

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, BasicTool, EztrimCutter,
                            FileInfo, Mux, Patch, RunnerConfig, SelfRunner,
                            VideoStream, X265Encoder)
from vardefunc.types import Range

core = vs.core



class Encoding:
    runner: SelfRunner

    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip
        assert self.file.a_src


        self.v_encoder = X265Encoder('magia_common/x265_settings')
        self.a_extracters = BasicTool('mkvextract', [self.file.path.to_str(), 'tracks', f'1:{self.file.a_src.set_track(1)}'])
        self.a_cutters = EztrimCutter(self.file, track=1)

    def run(self) -> None:
        assert self.file.a_src_cut

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC WEBRip by Vardë@Raws-Maji', JAPANESE),
                [AudioStream(self.file.a_src_cut.set_track(1), 'EAC3 2.0', JAPANESE)],
                None
            )
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            self.a_extracters, self.a_cutters, None,
            muxer
        )

        self.runner = SelfRunner(self.clip, self.file, config)
        self.runner.run()

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()
