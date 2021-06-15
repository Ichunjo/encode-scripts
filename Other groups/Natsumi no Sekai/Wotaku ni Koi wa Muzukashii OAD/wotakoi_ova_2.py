"""Wotakoi script"""
__author__ = 'Vardë'

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import havsfunc as hvf
import lvsfunc
import vapoursynth as vs
import vardefunc as vdf
from vardautomation import (FRENCH, AudioCutter, AudioEncoder, BasicTool,
                            EncodeGoBrr, FileInfo, MatroskaXMLChapters,
                            MplsReader, PresetAAC, PresetBD, VideoEncoder,
                            X265Encoder)
from vardefunc.mask import FreyChenG41
from vardefunc.misc import merge_chroma
from vardefunc.noise import AddGrain, Graigasm
from vsutil import depth, get_y, iterate, split

core = vs.core

core.num_threads = 16

NUM = __file__[-5:-3]


# MplsReader(Path('BD_VIDEO'), FRENCH, default_chap_name='Chapitre').write_playlist(Path().absolute())


JPBD = FileInfo(r'BD_VIDEO\BDMV\STREAM\00002.m2ts', 0, -24,
                preset=[PresetBD, PresetAAC])
JPBD.chapter = 'chapters/wotakoi_ova.xml'

CHAP_NAMES: List[Optional[str]] = ['Partie 1', 'Partie 1', 'ED']


ENCODER = X265Encoder('x265', Path('x265_settings'))

A_EXTRACTER = BasicTool('eac3to', [JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL'])
A_CUTTER = AudioCutter(JPBD, track=1)
A_ENCODER = AudioEncoder('qaac', Path('qaac_settings'), JPBD, track=1)


class EncodeGoBrrr(EncodeGoBrr):
    def __init__(self, clip: vs.VideoNode, file: FileInfo, /,
                 v_encoder: VideoEncoder,
                 a_extracters: Optional[Union[BasicTool, Sequence[BasicTool]]],
                 a_cutters: Optional[Union[AudioCutter, Sequence[AudioCutter]]],
                 a_encoders: Optional[Union[AudioEncoder, Sequence[AudioEncoder]]]) -> None:
        super().__init__(clip, file, v_encoder, v_lossless_encoder=None,
                         a_extracters=a_extracters, a_cutters=a_cutters, a_encoders=a_encoders)

    def run(self) -> None:
        self._parsing()
        self._encode()
        self._audio_getter()
        self.write_encoder_name_file('tags_aac.xml', 1)
        self.chapter()
        self.merge()

    def chapter(self) -> None:
        assert self.file.chapter
        chap = MatroskaXMLChapters(self.file.chapter)
        chap.set_names(CHAP_NAMES)

    def merge(self) -> None:
        """Merge function"""
        assert self.file.a_enc_cut
        assert self.file.chapter
        BasicTool('mkvmerge', [
            '-o', self.file.name_file_final,
            '--track-name', '0:HEVC BDRip by Vardë@Natsumi-no-Sekai', '--language', '0:jpn', self.file.name_clip_output,
            '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', self.file.a_enc_cut.format(1),
            '--chapter-language', 'fr', '--chapters', self.file.chapter
        ]).run()



class Filtering():
    def main(self) -> vs.VideoNode:
        """Vapoursynth filtering"""
        src = JPBD.clip_cut
        src = depth(src, 16)
        out = src


        denoise = hvf.SMDegrain(out, tr=2, thSAD=100)
        out = denoise




        y = get_y(out)

        lineart_mask = FreyChenG41().get_mask(y)
        lineart_mask = lineart_mask.std.Maximum().std.Convolution([1]*9)


        aa_strong = lvsfunc.aa.upscaled_sraa(y, 1.2)
        aa_weak = lvsfunc.aa.upscaled_sraa(y, 1.5, aafun=lvsfunc.aa.nnedi3())
        aa_clammp = lvsfunc.aa.clamp_aa(y, aa_weak, aa_strong, strength=3.5)
        aa_clammp = lvsfunc.misc.replace_ranges(aa_clammp, aa_strong, [(34046, src.num_frames-1)])

        aaa = merge_chroma(aa_clammp, out)
        out = aaa


        db_mask = self.deband_mask(out, kirsch_brz=(3000, 4500, 4500), rng_brz=(4000, 4000, 4000))

        deband = vdf.deband.dumb3kdb(out, 17, 49, grain=[48, 16])
        deband_a = vdf.deband.f3kbilateral(out, 20, 64, grain=[48, 16])


        deband = lvsfunc.misc.replace_ranges(deband, deband_a, [(6892, 7047)])
        deband = core.std.MaskedMerge(deband, out, db_mask)
        out = deband





        thrs = [x << 8 for x in (32, 80, 128, 176)]
        strengths = [(0.3, 0.1), (0.2, 0.1), (0.1, 0.0), (0.0, 0.0)]
        sizes = (1.2, 1.1, 1, 1)
        sharps = (70, 60, 50, 50)
        grainers = [AddGrain(seed=333, constant=False),
                    AddGrain(seed=333, constant=False),
                    AddGrain(seed=333, constant=True)]
        pref = iterate(get_y(out), core.std.Maximum, 2).std.Convolution([1]*9)
        grain = Graigasm(thrs, strengths, sizes, sharps, grainers=grainers).graining(out, prefilter=pref)
        out = grain


        decs = vdf.noise.decsiz(out, min_in=thrs[-2], max_in=thrs[-1])
        out = decs


        ref = denoise
        credits_maks = vdf.mask.luma_credit_mask(ref, 231 << 8).std.Convolution([1]*9)
        credit = out
        credit = lvsfunc.misc.replace_ranges(
            credit, core.std.MaskedMerge(credit, ref, credits_maks, 0),
            [(33806, src.num_frames-1)])
        out = credit



        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


    @staticmethod
    def deband_mask(clip: vs.VideoNode, kirsch_brz: Tuple[int, int, int], rng_brz: Tuple[int, int, int]) -> vs.VideoNode:
        prefilter = core.bilateral.Bilateral(clip, sigmaS=1.5, sigmaR=0.005)

        kirsch = vdf.mask.Kirsch().get_mask(prefilter).std.Binarize(kirsch_brz)
        rng = lvsfunc.mask.range_mask(prefilter, 3, 2).std.Binarize(rng_brz)
        kirsch, rng = [c.resize.Bilinear(format=vs.YUV444P16) for c in [kirsch, rng]]

        mask = core.std.Expr(split(kirsch) + split(rng), vdf.util.max_expr(6))

        return mask.rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = EncodeGoBrrr(filtered, JPBD, ENCODER, A_EXTRACTER, A_CUTTER, A_ENCODER)
    brrrr.run()
    brrrr.cleanup()
else:
    JPBD.clip_cut.set_output(0)

    FILTERED = Filtering().main()
    FILTERED.set_output(3)
