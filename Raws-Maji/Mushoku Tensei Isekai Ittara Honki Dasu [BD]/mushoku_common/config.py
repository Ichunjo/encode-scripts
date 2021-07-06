from typing import List, Union

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioCutter, AudioStream, BasicTool,
                            ChapterStream, FileInfo, MatroskaXMLChapters, Mux,
                            Patch, QAACEncoder, RunnerConfig, SelfRunner,
                            VideoStream, X265Encoder)
from vardautomation.types import Range

core = vs.core


class Encoding:
    runner: SelfRunner
    xml_tag: str = 'xml_tag.xml'
    do_chaptering: bool

    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip
        assert self.file.a_src


        self.v_encoder = X265Encoder('mushoku_common/x265_settings')
        self.a_extracters = [
            BasicTool('eac3to', [self.file.path.to_str(), '2:', self.file.a_src.format(1).to_str(), '-log=NUL'])
        ]
        self.a_cutters = [AudioCutter(self.file, track=1)]
        self.a_encoders = [QAACEncoder(self.file, track=1, xml_tag=self.xml_tag)]

    def run(self, *, do_chaptering: bool = True) -> None:
        assert self.file.a_enc_cut
        self.do_chaptering = do_chaptering

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC BDRip by Vardë@Raws-Maji', JAPANESE),
                [AudioStream(self.file.a_enc_cut.format(1), 'AAC 2.0', JAPANESE)],
                ChapterStream(self.file.chapter, JAPANESE) if do_chaptering and self.file.chapter else None
            )
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            self.a_extracters, self.a_cutters, self.a_encoders,
            muxer
        )


        self.runner = SelfRunner(self.clip, self.file, config)
        self.runner.run()

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()

    def cleanup(self) -> None:
        files = [self.xml_tag]
        if self.do_chaptering:
            assert self.file.chapter
            files.append(self.file.chapter)  # type: ignore

        self.runner.do_cleanup(*files)

    def chaptering(self, offset: int):
        assert self.file.chapter

        chap = MatroskaXMLChapters(self.file.chapter)
        chap.copy(self.file.name + '_tmp.xml')
        chap.shift_times(offset, self.clip.fps)
        chap.create_qpfile(self.file.qpfile, self.clip.fps)
        self.file.chapter = chap.chapter_file
