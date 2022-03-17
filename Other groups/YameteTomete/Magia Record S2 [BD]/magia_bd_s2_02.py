"""Magia Record S2 Script"""
import G41Fun as gf
import mvsfunc as mvf
import vapoursynth as vs
from debandshit import dumb3kdb
from lvsfunc.kernels import Bicubic, Spline36
# from debandshit.debanders import placebo_deband
from vardautomation import (ENGLISH, FileInfo2, MplsReader, PresetBD,
                            PresetBDWAV64, PresetChapXML, PresetOpus,
                            get_vs_core)
from vardautomation.config import FileInfo
from vardautomation.vpathlib import VPath
from vardefunc.aa import Eedi3SR, upscaled_sraa
from vardefunc.mask import MinMax, SobelStd, diff_creditless_mask, region_mask
from vardefunc.misc import DebugOutput, merge_chroma
from vardefunc.noise import Graigasm
from vardefunc.scale import to_444
from vardefunc.types import DuplicateFrame
from vardefunc.util import (finalise_output, initialise_input, remap_rfs,
                            replace_ranges)
from vsutil import depth, get_y, split
from vsutil.func import iterate

from magia_common import EncodingBluray, Mask, Thr, graigasm_args

core = get_vs_core(range(0, 24, 2))
NUM = __file__[-5:-3]


# MplsReader('[BDMV][211124][Magia Record S2][Vol.1]/BD', ENGLISH).write_playlist('chapters')

JPBD = FileInfo2(
    r'[BDMV][211124][Magia Record S2][Vol.1]\BD\BDMV\STREAM\00004.m2ts',
    (0, -50),
    preset=[PresetBDWAV64, PresetOpus, PresetChapXML]
)
# JPBD.name_clip_output = VPath('magia_bd_s2_02.265')
NCOP01 = FileInfo(
    r'[BDMV][211124][Magia Record S2][Vol.1]\BD\BDMV\STREAM\00008.m2ts',
    [(0, -24), DuplicateFrame(-25, 3)]
)
CRUNCHY = FileInfo(
    r'eps\[Élitiste] Magia Record - Puella Magi Madoka Magica Side Story Season 2 - The Eve of Awakening - 02.mp4',
    (-120, None)
)

# import lvsfunc

DEBUG = DebugOutput(
    source=JPBD.clip_cut + CRUNCHY.clip_cut,
    # NCOP=NCOP01.clip_cut
    # planes=lvsfunc.comparison.stack_planes(JPBD.clip_cut),
)

OPSTART, OPEND = 2590, 4747


class Filtering:
    @DEBUG.catch(op='@=')
    @finalise_output
    @initialise_input(bits=32)
    def main(self, src: vs.VideoNode = JPBD.clip_cut) -> vs.VideoNode:
        custom_mask1 = core.imwri.Read('masks/mask_1.png').resize.Point(format=vs.GRAYS, matrix=1)
        # out = depth(src, 32)
        out = src
        # debug = DEBUG

        full = to_444(out, None, None, True)
        luma = get_y(full)
        denoise_y = mvf.BM3D(luma, 0.85, radius1=1)
        denoise_y_b = mvf.BM3D(luma, 2.5, radius1=1)
        denoise_y = remap_rfs(
            denoise_y, core.std.MaskedMerge(denoise_y, denoise_y_b, custom_mask1),
            [(OPSTART+425, OPSTART+545), (OPSTART+603, OPSTART+691)]
        )
        denoise_y = core.expr.expr_cpp([denoise_y, luma], 'limitDenoise', source_path='magia_common/expr.cpp')
        assert out.format
        denoise_uv = core.ccd.CCD(full.resize.Point(format=vs.RGBS), 4).resize.Bicubic(
            format=out.format.id, matrix=1, filter_param_a=-0.5, filter_param_b=0.25
        )
        denoise = merge_chroma(denoise_y, denoise_uv)
        out = depth(denoise, 16)

        luma = get_y(out)

        fixborder = core.expr.expr_cpp(luma, 'fixBorder', source_path='magia_common/expr.cpp')
        fixborder = iterate(fixborder, core.std.Maximum, 5)
        fixborder = iterate(fixborder, core.std.Minimum, 2)

        aaa = upscaled_sraa(
            luma, 2,
            downscaler=Bicubic(-0.5, 0.25),
            # downscaler=Spline36(),
            singlerater=Eedi3SR(
                alpha=0.2, beta=0.4, gamma=400,
                mclip=SobelStd().get_mask(luma.resize.Bilinear(1920*2, 1080*2),
                                          lthr=4500, multi=1.75).std.Maximum().std.Minimum()
            )
        )
        aaa = remap_rfs(aaa, luma, [(31666, 33926), (OPSTART+425, OPSTART+545), (OPSTART+603, OPSTART+691)])
        aaa = remap_rfs(aaa, core.std.MaskedMerge(aaa, luma, self.op_mask().resize.Point(format=vs.GRAY16)),
                        (OPSTART, OPEND))
        out = merge_chroma(aaa, out)


        dehalo = gf.MaskedDHA(out, 1.25, 1.25, 0, 1.0)
        out = dehalo


        deband_mask = Mask().lineart_deband_mask(
            denoise.resize.Bilinear(format=vs.YUV444PS).rgsf.RemoveGrain(3),
            brz_rg=2200/65536, brz_ed=1700/65536, brz_ed_ret=12000/65536,
            ret_thrs=Thr(lo=(17 - 16) / 219, hi=(18 - 16) / 219)
        )
        deband_mask = core.expr.expr_cpp(
            split(deband_mask) + [get_y(out)], 'debandMask', vs.GRAY16,
            source_path='magia_common/expr.cpp'
        ).rgvs.RemoveGrain(3).rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)
        deband_mask = core.std.Lut(deband_mask, function=lambda x: x if x > 15000 else 0)


        db3k = dumb3kdb(out, 31, [40, 33], grain=0)
        db3k_masked = core.std.MaskedMerge(db3k, out, deband_mask)
        db3k_light = dumb3kdb(out, 25, [17, 17], grain=0)
        db3k_light_masked = core.std.MaskedMerge(db3k_light, out, deband_mask)
        db3kmore = dumb3kdb(out, 31, 49, grain=[15, 10])
        deband_a = core.std.MaskedMerge(
            db3kmore, db3k,
            MinMax._minmax(deband_mask, 10, core.std.Maximum).std.BoxBlur(0, 4, 4, 4, 4)
        )

        deband_b = core.std.MaskedMerge(deband_a, out, deband_mask)
        deband = core.expr.expr_cpp(
            [out, db3k_light_masked, db3k_masked, deband_b],
            ['debandLuma', 'debandChroma'], source_path='magia_common/expr.cpp'
        )
        deband = remap_rfs(
            deband, core.std.MaskedMerge(deband, db3kmore, custom_mask1.resize.Point(format=vs.GRAY16)),
            [(OPSTART+425, OPSTART+545), (OPSTART+603, OPSTART+691)]
        )
        out = deband

        grain = Graigasm(**graigasm_args).graining(out)  # type: ignore
        src = src.resize.Point(format=vs.YUV420P16)
        grain = core.std.MaskedMerge(grain, src, fixborder.resize.Point(format=vs.GRAY16))
        out = grain

        return out + self.endcard()

    @staticmethod
    def endcard() -> vs.VideoNode:
        clip = depth(CRUNCHY.clip_cut, 16)
        return gf.MaskedDHA(clip, 2.0, 2.0, 0, 1.0)

    @staticmethod
    def op_mask() -> vs.VideoNode:
        mask = diff_creditless_mask(
            JPBD.clip_cut,
            JPBD.clip_cut[OPSTART:OPEND].std.Convolution([1] * 9),
            NCOP01.clip_cut.std.Convolution([1] * 9),
            OPSTART,
            thr=100, prefilter=False
        )
        return mask


if __name__ == '__main__':
    # EncodingBluray(JPBD, Filtering().main(), NUM).run(upload_ftp=False)
    # EncodingBluray(JPBD, JPBD.clip_cut, NUM).run(upload_ftp=False)
    from vardautomation import Patch, X265Encoder
    Patch(
        X265Encoder('magia_common/x265_settings'),
        core.std.BlankClip(JPBD.clip_cut, format=vs.YUV420P10), JPBD, [(500, 600), (1000, 1200)]
    ).run()
    # brrr = EncodingBluray(JPBD, Filtering().main()[-120:], NUM).run()
    # from vardautomation import make_comps, PictureType, Writer
    # make_comps(
    #     dict(
    #         source=JPBD.clip_cut,
    #         filtered=Filtering().main(),
    #         encode=core.ffms2.Source('magia_bd_s2_01.mkv')
    #     ), 'comps', num=30, picture_types=[PictureType.P, PictureType.B],
    #     force_bt709=True, writer=Writer.OPENCV, slowpics=True,
    #     collection_name='Magia Record S201'
    # )
else:
    Filtering().main()
    # DEBUG <<= core.ffms2.Source('op.mkv')
    # DEBUG <<= Filtering.op_mask()
    # DEBUG <<= core.ffms2.Source('magia_bd_s2_01.mkv')
    pass
