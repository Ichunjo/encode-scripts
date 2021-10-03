"""YuYuYu Dai Mankai no Shou"""
from typing import Callable, List

import psutil
import vapoursynth as vs
from debandshit import dumb3kdb
from lvsfunc.kernels import Bicubic, Bilinear, Catrom, Mitchell
from vardautomation import FileInfo
from vardautomation.config import PresetEAC3, PresetWEB
from vardefunc.mask import detail_mask, diff_rescale_mask, region_mask
from vardefunc.misc import DebugOutput, merge_chroma
from vardefunc.noise import Graigasm
from vardefunc.scale import nnedi3_upscale
from vardefunc.util import finalise_output, remap_rfs
from vsmask.edge import FDOG
from vsutil import depth, get_y
from vsutil.clips import split

from yuyuyu_common import Denoise, EncodingWeb, graigasm_args, Scale

p_handle = psutil.Process()
p_handle.cpu_affinity(list(range(0, 24, 2)))
core = vs.core
core.num_threads = 12


NUM = __file__[-5:-3]

WEB_AMZ_VBR = FileInfo(
    f'eps/Yuuki Yuuna wa Yuusha de Aru - Dai Mankai no Shou - {NUM} (Amazon Prime VBR 1080p).mkv',
    (24, -24), preset=[PresetWEB, PresetEAC3]
)
WEB_BGLOBAL = FileInfo(
    f'eps/[NC-Raws] 结城友奈是勇者 大满开之章 - {NUM} [B-Global][WEB-DL][1080p][AVC AAC][ENG_TH_SRT][MKV].mkv'
)

OPSTART, OPEND = 32008, 34166

DEBUG = DebugOutput(WEB_AMZ_VBR.clip_cut)


@DEBUG.catch
@finalise_output
def filtering(debug: DebugOutput = DEBUG) -> vs.VideoNode:
    edgemask = FDOG().get_mask(get_y(WEB_BGLOBAL.clip_cut), 25, multi=2.25).std.Binarize(35)
    edgemask = edgemask.std.Maximum().std.Minimum().std.BoxBlur(0, 2, 2, 2, 2)
    src = core.std.MaskedMerge(WEB_BGLOBAL.clip_cut, WEB_AMZ_VBR.clip_cut, edgemask)

    _ef = [1, 0, 0]
    edgefix = core.edgefixer.ContinuityFixer(src, *[_ef] * 4)
    edgefix = remap_rfs(src, edgefix, [(222, 1081), (1888, 1930)])
    out = edgefix

    out = depth(out, 32)

    denoise = Denoise.bm3d(out, [1.25, 2, 2], radius=1)
    out = denoise


    descalers: List[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = [
        Bilinear().descale,
        Mitchell().descale,
        Bicubic(-.5, .25).scale,
    ]

    luma = get_y(out)
    descales = [descaler(luma, 1600, 900) for descaler in descalers]
    descale = core.std.Expr(descales, 'x y z min max y z max min z min')
    # upscale = Bicubic(-.5, .25).scale(nnedi3_upscale(descale), 1920, 1080)
    upscale = Bicubic(-.5, .25).scale(Scale.waifu2x(descale, 1), 1920, 1080)

    # upscale_s1 = Bicubic(-.5, .25).scale(nnedi3_upscale(core.descale.Debicubic(luma, 1280, 720, 1/3, 1/3)), 1920, 1080)
    upscale_s2 = Bicubic(-.5, .25).scale(nnedi3_upscale(core.descale.Debilinear(luma, 1280, 720)), 1920, 1080)
    upscale = remap_rfs(upscale, upscale_s2, [(222, 1081), (1888, 1930)])
    out = merge_chroma(upscale, out)

    out = depth(out, 16)

    credit = out
    ref = depth(src, 16)

    rescale_mask = diff_rescale_mask(ref.rgvs.RemoveGrain(3), 900, kernel=Catrom(), thr=65 * 256)
    rescale_mask_s2 = diff_rescale_mask(ref.rgvs.RemoveGrain(3), 720, kernel=Bilinear(), thr=75 * 256)
    rescale_mask_s2 = region_mask(rescale_mask_s2, 30, 30, 30, 30)
    # debug <<= dict(rescale_mask=rescale_mask, rescale_mask_s2=rescale_mask_s2)

    credit = remap_rfs(credit, ref, [(0, 222)])
    credit = remap_rfs(
        credit, core.std.MaskedMerge(credit, ref, rescale_mask),
        [(1517, 1888), (1952, 2039), (32186, 34166)]
    )
    credit = remap_rfs(
        credit, core.std.MaskedMerge(credit, ref, rescale_mask_s2),
        [(261, 322), (507, 1022), (1100, 1194), (1888, 1930)]
    )
    # 1517
    out = credit


    deband_mask = core.std.Expr(
        split(detail_mask(out, 3200, 5000).resize.Bilinear(format=vs.YUV444P16)),
        'x y z max max'
    ).rgvs.RemoveGrain(3)
    # debug <<= deband_mask
    deband = dumb3kdb(out, threshold=[33, 49], grain=24)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    # debug <<= deband
    out = deband

    grain = Graigasm(**graigasm_args).graining(out)  # type: ignore
    out = grain

    return out


# print(__name__)
if __name__ in ('__main__', '__vapoursynth__'):
    del DEBUG
    EncodingWeb(WEB_AMZ_VBR, filtering(), NUM, OPSTART, OPEND).run()
else:
    # filtering()
    pass
