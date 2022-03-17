from typing import Any, Dict, List, Optional, Tuple, Union

import vapoursynth as vs
from vardautomation import (
    ENGLISH, JAPANESE, AudioStream, ChapterStream, EztrimCutter, FFmpegAudioExtracter, FileInfo,
    FileInfo2, MKVAudioExtracter, Mux, OpusEncoder, Patch, RunnerConfig, SelfRunner, SoxCutter,
    VideoStream, X265Encoder
)
from vardautomation.tooling.audio import QAACEncoder
from vardefunc.types import Range

core = vs.core


class Encoding:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo, clip: vs.VideoNode, num: str) -> None:
        self.file = file
        self.clip = clip
        self.num = num
        self.v_encoder = X265Encoder('magia_common/x265_settings')

    def run(self, *, zones: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None, upload_ftp: bool = False) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut

        self.v_encoder = X265Encoder('magia_common/x265_settings', zones)

        a_extracter = MKVAudioExtracter(self.file, track_in=1, track_out=1)
        a_cutter = EztrimCutter(self.file, track=1)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC WEBRip by Vardë@Raws-Maji', JAPANESE),
                AudioStream(self.file.a_src_cut.set_track(1), 'EAC3 2.0', JAPANESE),
                None
            ), merge_args={'--ui-language': 'en'}
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            a_extracter, a_cutter, None,
            muxer,
            order=RunnerConfig.Order.AUDIO
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        if upload_ftp:
            runner.upload_ftp('YametoTomato', f'files/ongoing/magireco_s2/{self.num}/', ['--progress', '--sftp-set-modtime=false'])

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()


class EncodingBluray:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo2, clip: vs.VideoNode, num: str) -> None:
        self.file = file
        self.clip = clip
        self.num = num
        self.v_encoder = X265Encoder('magia_common/x265_settings')

    def run(self, *, zones: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None, upload_ftp: bool = False) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut
        assert self.file.a_enc_cut
        assert self.file.chapter

        self.v_encoder = X265Encoder('magia_common/x265_settings', zones)

        # a_extracter = FFmpegAudioExtracter(self.file, track_in=1, track_out=1)
        # a_cutter = SoxCutter(self.file, track=1)
        a_encoder = OpusEncoder(self.file, track=1, use_ffmpeg=True)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC BDRip by Vardë@Raws-Maji', JAPANESE),
                AudioStream(self.file.a_enc_cut.set_track(1), 'Opus 2.0', JAPANESE),
                ChapterStream(self.file.chapter, ENGLISH)
            ), merge_args={'--ui-language': 'en'}
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            None, None, a_encoder,
            muxer,
            order=RunnerConfig.Order.AUDIO
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        if upload_ftp:
            runner.upload_ftp('YametoTomato', f'files/ongoing/magireco_s2/{self.num}/', ['--progress', '--sftp-set-modtime=false'])

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()


class EncodingBlurayNC:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo2, clip: vs.VideoNode, num: str) -> None:
        self.file = file
        self.clip = clip
        self.num = num
        self.v_encoder = X265Encoder('magia_common/x265_settings')

    def run(self, *, zones: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut
        assert self.file.a_enc_cut

        self.v_encoder = X265Encoder('magia_common/x265_settings', zones)

        # a_extracter = FFmpegAudioExtracter(self.file, track_in=1, track_out=1)
        # a_cutter = SoxCutter(self.file, track=1)
        # a_encoder = QAACEncoder(self.file, track=1)
        a_encoder = OpusEncoder(self.file, track=1, use_ffmpeg=True)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC BDRip by Vardë@Raws-Maji', JAPANESE),
                # AudioStream(self.file.a_enc_cut.set_track(1), 'AAC 2.0', JAPANESE),
                AudioStream(self.file.a_enc_cut.set_track(1), 'Opus 2.0', JAPANESE),
                None
            ), merge_args={'--ui-language': 'en'}
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            None, None, a_encoder,
            muxer,
            order=RunnerConfig.Order.AUDIO
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()
