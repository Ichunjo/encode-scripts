from typing import Dict, List, Union

import vapoursynth as vs
from vardautomation import (FRENCH, JAPANESE, AudioStream, BlurayShow,
                            ChapterStream, FfmpegAudioExtracter, FileInfo, Mux,
                            Patch, PresetAAC, PresetBD, QAACEncoder,
                            RunnerConfig, SelfRunner, SoxCutter, VideoStream,
                            VPath, X265Encoder)
from vardautomation.tooling import EztrimCutter, MKVAudioExtracter
from vardefunc.types import Range

core = vs.core



class YuYuYuYu:
    def __init__(self) -> None:
        self.bdmv_folder = VPath('[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]')
        sub_bdmv_folder_1 = 'YUYUYU_YUUSYA_DISC1'
        self.sub_bdmv_folder_2 = 'YUYUYU_YUUSYA_DISC2'
        episodes: Dict[VPath, List[VPath]] = {}
        episodes[self.bdmv_folder / sub_bdmv_folder_1] = [
            VPath('BDMV/STREAM/00000.m2ts'),
            VPath('BDMV/STREAM/00001.m2ts'),
            VPath('BDMV/STREAM/00002.m2ts')
        ]
        episodes[self.bdmv_folder / self.sub_bdmv_folder_2] = [
            VPath('BDMV/STREAM/00000.m2ts'),
            VPath('BDMV/STREAM/00001.m2ts'),
            VPath('BDMV/STREAM/00002.m2ts')
        ]
        self.bd = BlurayShow(episodes, (0, -24), preset=[PresetBD, PresetAAC], lang=FRENCH)

    def episode(self, num: int, /) -> FileInfo:
        return self.bd.episode(num)

    @property
    def ncop(self) -> FileInfo:
        return FileInfo(
            self.bdmv_folder / self.sub_bdmv_folder_2 / 'BDMV/STREAM/00003.m2ts',
            (0, -24)
        )

    @property
    def nced(self) -> FileInfo:
        return FileInfo(
            self.bdmv_folder / self.sub_bdmv_folder_2 / 'BDMV/STREAM/00004.m2ts',
            (0, -24)
        )


class Encoding:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip
        self.v_encoder = X265Encoder('yuyuyu_common/x265_settings')

    def run(self) -> None:
        assert self.file.a_enc_cut
        assert self.file.chapter
        xml_file = self.file.workdir / 'tags_enc.xml'

        a_extracter = FfmpegAudioExtracter(self.file, track_in=1, track_out=1)
        a_cutter = SoxCutter(self.file, track=1)
        a_encoder = QAACEncoder(self.file, track=1, xml_tag=xml_file)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC BDRip by Vardë@Owlolf', JAPANESE),
                AudioStream(self.file.a_enc_cut.set_track(1), 'AAC 2.0', JAPANESE, xml_file),
                ChapterStream(self.file.chapter, FRENCH)
            ), merge_args={'--ui-language': 'en'}
        )

        config = RunnerConfig(
            self.v_encoder, None,
            a_extracter, a_cutter, a_encoder,
            muxer, RunnerConfig.Order.AUDIO
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        runner.cleanup_files.remove(self.file.name_clip_output)
        runner.cleanup_files.remove(self.file.chapter)
        runner.do_cleanup()
        runner.upload_ftp('OwlolfNAS', '/Owlolf-Animés/Yuyuyu S1 Blu-Ray', ['--sftp-set-modtime=false'])

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()


class EncodingWeb:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo, clip: vs.VideoNode, num: str, opstart: int, opend: int) -> None:
        self.file = file
        self.clip = clip
        self.num = num
        self.opstart = opstart
        self.opend = opend
        self.v_encoder = X265Encoder('yuyuyu_common/x265_settings_web')

    def run(self) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut

        part_1 = VPath('_audio_part_1.eac3')
        op = VPath('opening.eac3')
        part_2 = VPath('_audio_part_2.eac3')
        merge_audio = VPath('_merge.eac3')

        MKVAudioExtracter(self.file, track_in=1, track_out=1).run()
        EztrimCutter(self.file, track=1).run()

        EztrimCutter.ezpztrim(self.file.a_src_cut.set_track(1), part_1, (0, self.opstart), self.clip)
        EztrimCutter.ezpztrim(self.file.a_src_cut.set_track(1), part_2, (self.opend, None), self.clip)
        EztrimCutter.combine([part_1, op, part_2], merge_audio)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC WEBRip by Vardë@Owlolf', JAPANESE),
                AudioStream(merge_audio, 'EAC3 2.0', JAPANESE),
                None
            ), merge_args={'--ui-language': 'en'}
        )

        config = RunnerConfig(self.v_encoder, None, None, None, None, muxer)

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        runner.cleanup_files.remove(self.file.name_clip_output)
        runner.do_cleanup(self.file.a_src.set_track(1), self.file.a_src_cut.set_track(1), part_1, part_2, merge_audio)
        runner.upload_ftp('OwlolfNAS', f'/Owlolf-Animés/YuYuYu S3/{self.num}', ['--sftp-set-modtime=false'])

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()
