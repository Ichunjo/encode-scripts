from typing import Any, List, NamedTuple, Optional, Set, Tuple, cast

import EoEfunc as eoe
import vapoursynth as vs
from havsfunc import LSFmod
from lvsfunc.kernels import Bicubic
from lvsfunc.types import Range
from lvsfunc.util import replace_ranges
from vardefunc.aa import Eedi3SR, Nnedi3SS, upscaled_sraa
from vardefunc.mask import FDOG, Difference, ExLaplacian4, MinMax, SobelStd
from vardefunc.scale import to_444
from vardefunc.util import get_sample_type
from vsutil import Dither, depth, get_depth, get_y, iterate

core = vs.core



class Thr(NamedTuple):
    lo: float
    hi: float



PROPS_RGB: Set[Tuple[str, int]] = {
    ('_Matrix', 0)
}

PROPS_YUV_HD: Set[Tuple[str, int]] = {
    ('_Matrix', 1),
    ('_Transfer', 1),
    ('_Primaries', 1)
}


class SetFrameProp:
    @staticmethod
    def rgb(clip: vs.VideoNode) -> vs.VideoNode:
        for prop, val in PROPS_RGB:
            clip = clip.std.SetFrameProp(prop, intval=val)
        clip = clip.std.SetFrameProp('_Transfer', delete=True)
        clip = clip.std.SetFrameProp('_Primaries', delete=True)
        return clip

    @staticmethod
    def yuv_hd(clip: vs.VideoNode) -> vs.VideoNode:
        for prop, val in PROPS_YUV_HD:
            clip = clip.std.SetFrameProp(prop, intval=val)
        return clip


class Denoise:
    @staticmethod
    def bm3d(clip: vs.VideoNode, sigma: List[float], radius: int, **kwargs: Any) -> vs.VideoNode:
        return cast(vs.VideoNode, eoe.denoise.BM3D(clip, sigma, radius=radius, CUDA=True, **kwargs))


class Scale:
    @staticmethod
    def to_444(c: vs.VideoNode) -> vs.VideoNode:
        return cast(vs.VideoNode, to_444(c, znedi=False))

    @staticmethod
    def to_yuv420(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        assert ref.format
        return clip.resize.Bicubic(
            format=ref.format.id, matrix=1, dither_type=Dither.ERROR_DIFFUSION,
            filter_param_a_uv=-0.5, filter_param_b_uv=0.25
        )


class AA:
    @staticmethod
    def upscaled_sraa(clip: vs.VideoNode, rfactor: float = 2.0, rep: int = 13, contrasharp: float = 20.0, **kwargs: Any) -> vs.VideoNode:
        eedi3_args = dict(eedi3cl=False, gamma=250, nrad=1, mdis=15)
        eedi3_args |= kwargs
        aaa = upscaled_sraa(
            clip, rfactor=rfactor,
            supersampler=Nnedi3SS(opencl=False, nns=2),
            downscaler=Bicubic(-0.5, 0.25),
            singlerater=Eedi3SR(**eedi3_args)
        ).rgvs.Repair(clip, rep)
        return LSFmod(aaa, strength=contrasharp, Smode=3, edgemode=0, source=clip)


class Mask:
    class ExLaplaDOG(ExLaplacian4):
        def __init__(self, *, ret: bool = False) -> None:
            self.ret = ret
            super().__init__()

        def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
            assert clip.format
            if self.ret:
                pre = depth(clip, 16).retinex.MSRCP(sigma=[25, 150, 280], upper_thr=9e-4)
                pre = pre.resize.Point(format=clip.format.id)
            else:
                pre = clip

            exlaplacian4 = super()._compute_mask(pre)
            fdog = FDOG().get_mask(pre)

            mask = core.std.Expr((exlaplacian4, fdog), 'x y max')
            mask = mask.std.Crop(right=2).resize.Point(mask.width, src_width=mask.width)

            return mask

    def lineart_deband_mask(self, clip: vs.VideoNode,
                            brz_rg: float, brz_ed: float, brz_ed_ret: float,
                            ret_thrs: Thr, extra: bool = True) -> vs.VideoNode:
        range_mask = MinMax(6, 0).get_mask(clip).std.Binarize(brz_rg)
        # edgemask = self.ExLaplaDOG().get_mask(clip).std.Binarize(brz_ed)
        edgemask = SobelStd().get_mask(clip).std.Binarize(brz_ed)
        edgemaskret = self.ExLaplaDOG(ret=True).get_mask(clip).std.Binarize(brz_ed_ret)

        # Keep retinex edgemask only under th_lo
        th_lo, th_hi = ret_thrs
        strength = f'{th_hi} x - {th_hi} {th_lo} - /'
        edgemask = core.std.Expr(
            [clip.std.BoxBlur(0, 3, 3, 3, 3), edgemask, edgemaskret],
            f'x {th_lo} > x {th_hi} < and z ' + strength + ' * y 1 ' + strength + f' - * + x {th_lo} <= z y ? ?'
        )

        lmask = core.std.Expr((range_mask, edgemask), 'x y max')

        if extra:
            lmask = lmask.rgsf.RemoveGrain(22).rgsf.RemoveGrain(11)
            lmask = iterate(lmask, core.std.Inflate, 4)

        return lmask


    @staticmethod
    def limit_grain_mask(clip: vs.VideoNode, lthr: float = 16 << 8, hthr: float = 20 << 8) -> vs.VideoNode:
        bits = get_depth(clip)
        is_float = get_sample_type(clip) == vs.FLOAT
        peak = 1.0 if is_float else (1 << bits) - 1

        def func(x: float) -> int:
            if x <= lthr:
                x = peak
            elif x > hthr:
                x = 0.0
            else:
                x = abs(((x - lthr) / (hthr - lthr)) * peak - peak) ** 0.75
            return round(x)

        return core.std.Lut(
            get_y(clip).resize.Bilinear(960, 540).std.BoxBlur(0, 2, 2, 2, 2),
            function=func
        ).resize.Bilinear(1920, 1080)


    @staticmethod
    def restore_credits(clip: vs.VideoNode, ref: vs.VideoNode,
                        oprange: Optional[Tuple[int, int]],
                        edrange: Optional[Tuple[int, int]],
                        **creditless_args: Any) -> vs.VideoNode:
        rng: List[Range] = []
        if oprange:
            rng += [oprange]
            opstart, opend = oprange
        else:
            opstart, opend = (None, ) * 2
        if edrange:
            rng += [edrange]
            edstart, edend = edrange
        else:
            edstart, edend = (None, ) * 2

        mask = Difference().creditless_oped(opstart=opstart, opend=opend, edstart=edstart, edend=edend, **creditless_args)

        return replace_ranges(clip, core.std.MaskedMerge(clip, ref, mask), rng)
