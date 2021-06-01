# noqa
import os
from typing import Optional, Sequence, Union

import vapoursynth as vs
from vardautomation import (AudioCutter, AudioEncoder, BasicTool, EncodeGoBrr,
                            FileInfo, LosslessEncoder, Properties,
                            VideoEncoder)


class EncodeGoBrrr(EncodeGoBrr):
    def __init__(self, clip: vs.VideoNode, file: FileInfo, /,
                 v_encoder: VideoEncoder, v_lossless_encoder: Optional[LosslessEncoder],
                 a_extracters: Optional[Union[BasicTool, Sequence[BasicTool]]] = None,
                 a_cutters: Optional[Union[AudioCutter, Sequence[AudioCutter]]] = None,
                 a_encoders: Optional[Union[AudioEncoder, Sequence[AudioEncoder]]] = None) -> None:
        super().__init__(clip, file, v_encoder, v_lossless_encoder, a_extracters, a_cutters, a_encoders)

        # self.temp_chap = self.file.chapter + '_chap_temp.txt'

        # if isinstance(chap_names, Sequence):
        #     self.chap_names = chap_names
        # else:
        #     raise ValueError('Set chap_names moron')

    def run(self) -> None:
        self._parsing()
        self._encode()
        self._audio_getter()
        self._set_encoder_name()
        # self.chapter()
        self.merge()

    def chapter(self) -> None:
        pass

    def merge(self) -> None:
        assert self.file.a_enc_cut
        assert self.file.chapter
        BasicTool('mkvmerge', [
            '-o', self.file.name_file_final,
            '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', self.file.name_clip_output,
            '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', self.file.a_enc_cut.format(1),
            # '--chapter-language', 'jpn', '--chapters', self.file.chapter
        ]).run()

    def _set_encoder_name(self) -> None:
        assert self.file.a_enc_cut
        aac_encoder_name = Properties.get_encoder_name(self.file.a_enc_cut.format(1))
        with open('tags_aac.xml', 'w') as f:  # noqa
            f.writelines(
                ['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                 '<Simple>', '<Name>ENCODER</Name>', f'<String>{aac_encoder_name}</String>', '</Simple>',
                 '</Tag>', '</Tags>'])

    def cleanup(self) -> None:  # noqa
        self.file.cleanup()
        os.remove('tags_aac.xml')





# class EncodeGoBrrr():  # noqa
#     def __init__(self: EncodeGoBrrr, clip: vs.VideoNode, file: FileInfoMore) -> None:
#         self.clip = clip
#         self.file = file

#     def run(self: EncodeGoBrrr) -> None:  # noqa
#         self._parsing()
#         self._encode()
#         self._audio_getter()
#         self._get_tag_aac()
#         self._merge()

#     def _parsing(self: EncodeGoBrrr) -> None:
#         parser = Parser(self.file)
#         self.file, self.clip = parser.parsing(self.file, self.clip)

#     def _encode(self: EncodeGoBrrr) -> None:
#         if self.file.do_lossless:
#             NvencEncoder('C:/NVEncC_5.30_x64/NVEncC64.exe', Path('gotoubun_common/nvenc_settings'),
#                          self.clip, self.file, progress_update=None)
#             self.clip = core.lsmas.LWLibavSource(self.file.name_clip_output_lossless)

#         if not Path(self.file.name_clip_output).exists():
#             X265Encoder('x265', Path('gotoubun_common/x265_settings'), self.clip, self.file,
#                         progress_update=lambda v, e:
#                             print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end=""))

#     def _audio_getter(self: EncodeGoBrrr) -> None:
#         assert self.file.a_src
#         assert self.file.a_src_cut
#         assert self.file.a_enc_cut

#         if not Path(self.file.a_src).exists():
#             BasicTool('eac3to', [self.file.src, '2:', self.file.a_src, '-log=NUL'])

#         if not Path(self.file.a_src_cut).exists():
#             eztrim(self.file.clip, (self.file.frame_start, self.file.frame_end), self.file.a_src, self.file.a_src_cut)

#         if not Path(self.file.a_enc_cut).exists():
#             AudioEncoder('qaac', Path('gotoubun_common/qaac_settings'), self.file)

#     def _get_tag_aac(self: EncodeGoBrrr) -> None:
#         ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder',
#                         '-print_format', 'default=nokey=1:noprint_wrappers=1', self.file.a_enc_cut]
#         encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')

#         with open('tags_aac.xml', 'w') as f:  # noqa
#             f.writelines(
#                 ['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
#                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
#                  '</Tag>', '</Tags>'])

#     def _merge(self: EncodeGoBrrr) -> None:
#         assert self.file.a_enc_cut
#         assert self.file.chapter
#         BasicTool('mkvmerge', [
#             '-o', self.file.name_file_final,
#             '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', self.file.name_clip_output,
#             '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', self.file.a_enc_cut,
#             # '--chapter-language', 'jpn', '--chapters', self.file.chapter
#         ])

#     def cleanup(self: EncodeGoBrrr, **kwargs: Dict[str, bool]) -> None:  # noqa
#         self.file.cleanup(**kwargs)
