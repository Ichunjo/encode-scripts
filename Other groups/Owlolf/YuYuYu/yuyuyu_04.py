"""YuYuYu script"""

from typing import Any, Dict, Tuple, cast
from debandshit.debanders import placebo_deband

import vapoursynth as vs
import yaml
from debandshit import dumb3kdb
from G41Fun import MaskedDHA
from lvsfunc.kernels import Bicubic, Mitchell
from muvsfunc import SSIM_downsample
from vardautomation import (FRENCH, FileInfo, MplsReader, PresetAAC, PresetBD,
                            VPath)
from vardefunc.aa import Eedi3SR, Nnedi3SS, upscaled_sraa
from vardefunc.mask import (FDOG, detail_mask, diff_creditless_mask,
                            diff_rescale_mask, region_mask)
from vardefunc.misc import DebugOutput, merge_chroma
from vardefunc.noise import Graigasm
from vardefunc.scale import nnedi3_upscale, to_444
from vardefunc.util import finalise_output, remap_rfs
from vsutil import depth, get_y, split

from yuyuyu_common import Denoise, Encoding, graigasm_args

core = vs.core

NUM = int(__file__[-5:-3])

with open('yuyuyu_m2ts_episode.yml', 'r', encoding='utf-8') as config:
    paths = cast(Dict[str, Any], yaml.load(config, yaml.Loader))

JPBD = FileInfo(VPath(*paths['first_press'][NUM-1]), (0, -24), preset=[PresetBD, PresetAAC], workdir='__workdir')
JPBD2 = FileInfo(VPath(*paths['collector'][NUM-1]), (0, -24), preset=[PresetBD])
JPBD_NCOP = FileInfo(VPath(*paths['ncops'][1]), (0, -24), preset=[PresetBD])
# JPBD_NCED = FileInfo(VPath(*paths['nceds'][1]), (0, -24), preset=[PresetBD])

BDMV_FOLDER = VPath(paths['first_press_path'], paths['first_press_vol2'])

CHAPTER_FOLDER = BDMV_FOLDER / 'chapters'
CHAPTER_FOLDER.mkdir(parents=True, exist_ok=True)
if not sorted(CHAPTER_FOLDER.glob('*')):
    MplsReader(BDMV_FOLDER, FRENCH, 'Chapitre').write_playlist(CHAPTER_FOLDER)

JPBD.chapter = CHAPTER_FOLDER / f'00000_{JPBD.path.stem}.xml'

OPSTART, OPEND = 1296, 3453
EDSTART, EDEND = 31264, 34812

# import lvsfunc

DEBUG = DebugOutput(
    JPBD.clip_cut,
    # JPBD2.clip_cut
    # lvsfunc.comparison.stack_planes(depth(JPBD.clip_cut, 32)),
    # lvsfunc.comparison.stack_planes((depth(JPBD2.clip_cut, 32)))
    # JPBD_NCOP.clip_cut
    # lvsfunc.comparison.stack_planes(depth(JPBD.clip_cut, 32))
)


class Filtering:
    clip: vs.VideoNode

    def __init__(self) -> None:
        self.clip = self.main()

    @DEBUG.catch(op='@=')
    @finalise_output
    # @initialise_input
    def main(self, debug: DebugOutput = DEBUG) -> vs.VideoNode:
        mean = core.average.Mean([depth(c, 32) for c in [JPBD.clip_cut, JPBD2.clip_cut[:35173]]], preset=3)
        mean = merge_chroma(depth(JPBD.clip_cut, 32), mean)
        out = mean.std.SetFrameProp('_Matrix', intval=1)

        ccd = self.ccd(out, 5)
        den_y = Denoise.bm3d(get_y(out), [0.8, 0, 0], radius=1, profile='fast', ref=get_y(ccd))
        denoise = merge_chroma(den_y, ccd)
        out = denoise

        pre = get_y(out)
        descale = core.descale.Debicubic(pre, 1280, 720, 1/3, 1/3)
        up1 = nnedi3_upscale(descale)

        mclip = FDOG().get_mask(up1).std.BinarizeMask(5500/65535)
        # debug[10] = mclip

        uspcale = upscaled_sraa(
            up1, 1.2, 1920, 1080, downscaler=self.SSIMDownscaler(0, 0),
            supersampler=Nnedi3SS(opencl=True, nsize=0, nns=2),
            singlerater=Eedi3SR(eedi3cl=False, nnedi3cl=True, beta=0.6, gamma=200, nrad=1, mdis=15, mclip=mclip, nnedi3_args=dict(nns=2))
        )
        out = merge_chroma(uspcale, out)
        out = depth(out, 16)

        dehalo = MaskedDHA(out, 1.25, 1.25, 0, 1.)
        out = dehalo

        rescale_mask = diff_rescale_mask(pre, kernel=Mitchell(), thr=75/219)
        rescale_mask = region_mask(rescale_mask, 30, 30, 30, 30).std.Expr('x 65535 *', vs.GRAY16)
        src = depth(mean, 16)
        ncop_mask = diff_creditless_mask(
            out, src[OPSTART:OPEND], JPBD_NCOP.clip_cut, OPSTART,
            140 << 8, expand=2, prefilter=True
        )
        # nced_mask = diff_creditless_mask(
        #     out, src[EDSTART:EDEND], JPBD_NCED.clip_cut, EDSTART,
        #     140 << 8, expand=2, prefilter=True
        # )
        # nc_mask = core.std.Expr((ncop_mask, nced_mask), 'x y +')
        ncop_mask = ncop_mask.std.Convolution([1]*9)

        # debug <<= ncop_mask
        # debug <<= rescale_mask
        credit = out
        ref = depth(denoise, 16)
        credit = remap_rfs(
            credit, core.std.MaskedMerge(credit, ref, rescale_mask),
            [(30965, 32745), (14393, 14435), (14481, 14530), (14607, 14667),
             (15179, 15240)]
        )
        # credit = remap_rfs(credit, core.std.MaskedMerge(credit, ref, nc_mask), [(OPSTART, OPEND), (EDSTART, EDEND)])
        credit = remap_rfs(credit, core.std.MaskedMerge(credit, ref, ncop_mask), [(OPSTART, OPEND)])
        out = credit


        deband_mask = core.std.Expr(
            split(detail_mask(out, 3200, 5000).resize.Bilinear(format=vs.YUV444P16)),
            'x y z max max'
        ).rgvs.RemoveGrain(3)
        # debug <<= deband_mask
        deband = dumb3kdb(out, threshold=36, grain=24)
        # deband_b = dumb3kdb(out, threshold=64, grain=24)
        deband_b = placebo_deband(out, threshold=6.5, iterations=3)
        deband = remap_rfs(deband, deband_b, [(34503, 34519)])
        deband = core.std.MaskedMerge(deband, out, deband_mask)
        # debug <<= deband
        out = deband

        grain = Graigasm(**graigasm_args).graining(out)  # type: ignore
        out = grain

        # debug <<= lvsfunc.comparison.stack_planes(depth(out, 32))

        return out

    def ccd(self, clip: vs.VideoNode, thr: float) -> vs.VideoNode:
        upscale = to_444(clip, None, None, True, False).resize.Bicubic(format=vs.RGBS)
        return core.ccd.CCD(upscale, thr).resize.Bicubic(
            format=clip.format.id, matrix=1, filter_param_a=-0.5, filter_param_b=0.25  # type: ignore
        )

    class SSIMDownscaler(Bicubic):
        def scale(self, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float] = (0, 0)) -> vs.VideoNode:
            return SSIM_downsample(clip, width, height, smooth=((3 ** 2 - 1) / 12) ** 0.5,
                                   sigmoid=True, filter_param_a=self.b, filter_param_b=self.c, **self.kwargs)



if __name__ == '__main__':
    del DEBUG
    Encoding(JPBD, Filtering().clip).run()
else:
    outclip = Filtering().clip
    # DEBUG <<= lvsfunc.comparison.stack_planes(depth(JPBD.clip_cut, 32))
    # DEBUG <<= lvsfunc.comparison.stack_planes(depth(outclip, 32))
    pass
