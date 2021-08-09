"""Magia Record S2 Script"""
import vapoursynth as vs
from debandshit import dumb3kdb, placebo_deband
from vardautomation import FileInfo, PresetEAC3, PresetWEB
from vardefunc.aa import Eedi3SR
from vardefunc.mask import MinMax, SobelStd
from vardefunc.misc import DebugOutput, merge_chroma
from vardefunc.noise import Graigasm
from vardefunc.types import DuplicateFrame as DF
from vardefunc.util import finalise_output, initialise_input, replace_ranges
from vsutil import depth, get_y, split

from magia_common import Denoise, Encoding, Mask, Thr, graigasm_args

core = vs.core


NUM = __file__[-5:-3]


WEB_CRU = FileInfo(
    f'eps/Magia Record S2 - {NUM} (Crunchyroll 1080p).mkv',
    preset=PresetWEB
)
WEB_AMZ_CBR = FileInfo(
    f'eps/Magia Record S2 - {NUM} (Amazon Prime CBR 1080p).mkv',
    None,
    preset=(PresetWEB, PresetEAC3)
)
WEB_BIL = FileInfo(
    f'eps/[NC-Raws] 魔法纪录 魔法少女小圆外传 第二季 －觉醒前夜－ - {NUM} [B-Global][WEB-DL][2160p][AVC AAC][CHS_CHT_ENG_TH_SRT][MKV].mkv',
    None,
    preset=PresetWEB
)
DEBUG = DebugOutput(
    # CR=WEB_CRU.clip_cut,
    # Bilibili=WEB_BIL.clip_cut.resize.Bicubic(1920, 1080, format=vs.YUV444P8),
    # AMZ_CBR=WEB_AMZ_CBR.clip_cut,
    num=9, props=7
)

# AMAZON TRIMS
WEB_AMZ_CBR.trims_or_dfs = [
    # Video trims
    # DF(0, 2), (0, 5427), DF(5426, 3), (5427, 14453), DF(14452, 1), DF(14453, 2), (14453, None)
    # Audio trims | NOTE: Remove 1 frame because of some delay in eac3 codec
    DF(0, 2), (1, 14453), DF(14452, 1), DF(14453, 2), (14453, None)
]
# AMAZON TRIMS | fill for the endcard
WEB_AMZ_CBR.trims_or_dfs += [
    DF(0, WEB_CRU.clip_cut.num_frames - WEB_AMZ_CBR.clip_cut.num_frames)
]
# BILIBILI TRIMS
WEB_BIL.trims_or_dfs = [
    (0, 4197), DF(4434, 14), (4434, 14681), DF(14680, 1), (14681, None),
]
WEB_BIL.trims_or_dfs += [
    DF(0, WEB_CRU.clip_cut.num_frames - WEB_BIL.clip_cut.num_frames)
]


DEBUG <<= dict(
    CR=WEB_CRU.clip_cut,
    Bilibili=WEB_BIL.clip_cut.resize.Bicubic(1920, 1080, format=vs.YUV444P8),
)


class Filtering:
    @DEBUG.catch(op='@=')
    @finalise_output
    @initialise_input(bits=32)
    def main(self, src: vs.VideoNode = WEB_BIL.clip_cut) -> vs.VideoNode:
        # debug = DEBUG
        pre = get_y(src)
        aaa = Eedi3SR(eedi3cl=True, gamma=250, nrad=1, mdis=15).do_aa()(pre)
        aaa = core.std.MaskedMerge(pre, aaa, SobelStd().get_mask(pre))
        aaa = merge_chroma(aaa, src)
        out = aaa


        endcard = replace_ranges(out, WEB_CRU.clip_cut, [(33903, None)], mismatch=True)
        out = endcard

        rescale = out.resize.Bicubic(
            1920, 1080, vs.YUV444PS,
            filter_param_a=-0.5, filter_param_b=0.25,
            filter_param_a_uv=0, filter_param_b_uv=0.5
        )
        out = rescale

        denoise = Denoise.bm3d(out, [1.0, 1.5, 1.5], radius=1, profile='fast')
        out = denoise

        out = depth(out, 16)

        deband_mask = Mask().lineart_deband_mask(
            denoise.rgsf.RemoveGrain(3), brz_rg=2200/65536, brz_ed=1700/65536, brz_ed_ret=12000/65536,
            ret_thrs=Thr(lo=(17 - 16) / 219, hi=(18 - 16) / 219)
        )
        deband_mask = core.std.Expr(
            split(deband_mask) + [get_y(rescale)],
            f'a {(18-16)/219} > x y z max max 65535 * 0 ?', vs.GRAY16
        ).rgvs.RemoveGrain(3).rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)
        deband_mask = core.std.Lut(deband_mask, function=lambda x: x if x > 15000 else 0)
        # debug[10] = deband_mask
        # debug <<= MinMax._minmax(deband_mask, 10, core.std.Maximum).std.BoxBlur(0, 4, 4, 4, 4)

        db3k = dumb3kdb(out, 20, [45, 40], grain=30)
        db3k_masked = core.std.MaskedMerge(db3k, out, deband_mask)
        dbpl = placebo_deband(out, 18, 7.5, grain=2.0)
        deband_a = core.std.MaskedMerge(
            dbpl, db3k,
            MinMax._minmax(deband_mask, 10, core.std.Maximum).std.BoxBlur(0, 4, 4, 4, 4)
        )
        deband_b = core.std.MaskedMerge(deband_a, out, deband_mask)
        deband = core.std.Expr(
            [deband_b, out, db3k_masked],
            [f'y {30<<8} < z x ?', 'z']
        )
        out = deband

        grain = Graigasm(**graigasm_args).graining(out)  # type: ignore
        out = grain

        return out


if __name__ == '__main__':
    vs.clear_outputs()
    FINAL_FILE = WEB_BIL
    FINAL_FILE.path = WEB_AMZ_CBR.path
    FINAL_FILE.a_src = WEB_AMZ_CBR.a_src
    FINAL_FILE.a_src_cut = WEB_AMZ_CBR.a_src_cut
    FINAL_FILE._trims_or_dfs = WEB_AMZ_CBR._trims_or_dfs
    Encoding(FINAL_FILE, Filtering().main()).run()
else:
    # DEBUG <<= dict(CR=WEB_CRU.clip_cut, Bilibili=WEB_BIL.clip_cut.resize.Bicubic(1920, 1080, format=vs.YUV444P8), Funi=WEB_FUN.clip_cut)
    Filtering().main()
    # DEBUG[20] = lvsfunc.comparison.stack_planes(to_444(depth(WEB_CRU.clip_cut, 32), None, None, True, False))
    # DEBUG[21] = lvsfunc.comparison.stack_planes(WEB_BIL.clip_cut.resize.Bicubic(1920, 1080, format=vs.YUV444PS))
