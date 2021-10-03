"""WaSuYU TV script"""

from typing import Tuple

import vapoursynth as vs
from debandshit import dumb3kdb
from lvsfunc.kernels import Bicubic, Bilinear
from muvsfunc import SSIM_downsample
from vardautomation import (FRENCH, FileInfo, MplsReader, PresetAAC, PresetBD,
                            VPath)
from vardautomation.shaders import Shader
from vardefunc.aa import Eedi3SR, Nnedi3SS, upscaled_sraa
from vardefunc.mask import (FDOG, detail_mask, diff_creditless_mask,
                            diff_rescale_mask, region_mask)
from vardefunc.misc import DebugOutput, merge_chroma
from vardefunc.noise import Graigasm
from vardefunc.scale import fsrcnnx_upscale, to_444
from vardefunc.util import finalise_output, initialise_input, remap_rfs
from vsutil import depth, get_y, split

from yuyuyu_common import Denoise, Encoding, graigasm_args

core = vs.core
core.num_threads = 14

NUM = int(__file__[-5:-3])

BDMV_FOLDER = VPath('[BDMV][171220][Yuuki Yuuna wa Yuusha de Aru Washio Sumi no Shou][BD-BOX]')
SUB_BDMV_FOLDER = 'YUYUYU_WASHIO_DISC2'
CHAPTER_FOLDER = BDMV_FOLDER / SUB_BDMV_FOLDER / 'chapters'
CHAPTER_FOLDER.mkdir(parents=True, exist_ok=True)

if not sorted(CHAPTER_FOLDER.glob('*')):
    MplsReader(BDMV_FOLDER / SUB_BDMV_FOLDER, FRENCH, 'Chapitre').write_playlist(CHAPTER_FOLDER)

JPBD = FileInfo(
    BDMV_FOLDER / SUB_BDMV_FOLDER / f'BDMV/STREAM/0000{(NUM - 1) % 3}.m2ts',
    (0, -24), preset=[PresetBD, PresetAAC], workdir='__workdir'
)
JPBD.chapter = CHAPTER_FOLDER / f'00000_0000{(NUM - 1) % 3}.xml'

JPBD_NCOP = FileInfo(
    BDMV_FOLDER / SUB_BDMV_FOLDER / 'BDMV/STREAM/00004.m2ts',
    (0, -24)
)
JPBD_NCED = FileInfo(
    BDMV_FOLDER / SUB_BDMV_FOLDER / 'BDMV/STREAM/00006.m2ts',
    (0, -24)
)
OPSTART, OPEND = 1726, 3885
EDSTART, EDEND = 32656, 34814


# import lvsfunc

DEBUG = DebugOutput(
    JPBD.clip_cut,
    # JPBD_NCOP.clip.std.BlankClip(length=OPSTART) + JPBD_NCOP.clip_cut,
    # JPBD_NCED.clip.std.BlankClip(length=EDSTART) + JPBD_NCED.clip_cut,
    # lvsfunc.comparison.stack_compare(JPBD.clip_cut, JPBD_NCOP.clip.std.BlankClip(length=OPSTART) + JPBD_NCOP.clip_cut, height=486),
    # lvsfunc.comparison.stack_compare(JPBD.clip_cut, JPBD_NCED.clip.std.BlankClip(length=EDSTART) + JPBD_NCED.clip_cut, height=486),
    # lvsfunc.comparison.stack_planes(depth(JPBD.clip_cut, 32))
)


class Filtering:
    clip: vs.VideoNode

    def __init__(self) -> None:
        self.clip = self.main()

    @DEBUG.catch(op='@=')
    @finalise_output
    @initialise_input
    def main(self, src: vs.VideoNode = JPBD.clip_cut, debug: DebugOutput = DEBUG) -> vs.VideoNode:
        _ef = [1, 0, 0]
        edgefix = core.edgefixer.ContinuityFixer(src, *[_ef] * 4)
        out = edgefix

        out = depth(out, 32)

        ccd = self.ccd(out, 5)
        den_y = Denoise.bm3d(get_y(out), [0.8, 0, 0], radius=1, profile='fast', ref=get_y(ccd))
        denoise = merge_chroma(den_y, ccd)
        out = denoise

        pre = get_y(out)
        descale = core.descale.Debilinear(pre, 1280, 720)
        up1 = fsrcnnx_upscale(
            descale, height=descale.height*2, shader_file=Shader.FSRCNNX_56_16_4_1,
            strength=80, overshoot=1.5, undershoot=1.1
        )

        mclip = FDOG().get_mask(up1).std.BinarizeMask(5500/65535)
        # debug[10] = mclip

        uspcale = upscaled_sraa(
            up1, 1.2, 1920, 1080, downscaler=self.SSIMDownscaler(0, 0),
            supersampler=Nnedi3SS(opencl=True, nsize=0, nns=2),
            singlerater=Eedi3SR(eedi3cl=False, nnedi3cl=True, beta=0.6, gamma=200, nrad=1, mdis=15, mclip=mclip, nnedi3_args=dict(nns=2))
        )
        out = merge_chroma(uspcale, out)
        out = depth(out, 16)


        # rescale_mask = diff_rescale_mask(pre, kernel=Bilinear(), thr=30/219)
        # rescale_mask = region_mask(rescale_mask, 30, 30, 30, 30).std.Expr('x 65535 *', vs.GRAY16)
        # debug <<= rescale_mask
        ncop_mask = diff_creditless_mask(
            out, src[OPSTART:OPEND], JPBD_NCOP.clip_cut, OPSTART,
            130 << 8, expand=2, prefilter=True
        )
        nced_mask = diff_creditless_mask(
            out, src[EDSTART:EDEND], JPBD_NCED.clip_cut, EDSTART,
            130 << 8, expand=2, prefilter=True
        )
        nc_mask = core.std.Expr((ncop_mask, nced_mask), 'x y +')
        nc_mask = nc_mask.std.Convolution([1]*9)

        # debug <<= rescale_mask
        # debug <<= nc_mask
        credit = out
        ref = depth(denoise, 16)
        # credit = remap_rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask), [(24566, 25910)])
        credit = remap_rfs(credit, core.std.MaskedMerge(credit, ref, nc_mask), [(OPSTART, OPEND), (EDSTART, EDEND)])
        out = credit


        deband_mask = core.std.Expr(
            split(detail_mask(out, 3200, 5000).resize.Bilinear(format=vs.YUV444P16)),
            'x y z max max'
        ).rgvs.RemoveGrain(3)
        # debug <<= deband_mask
        deband = dumb3kdb(out, threshold=26, grain=24)
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
    Filtering()
    pass
