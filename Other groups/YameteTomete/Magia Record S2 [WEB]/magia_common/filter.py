from typing import Any, List, NamedTuple, cast

import EoEfunc as eoe
import vapoursynth as vs
from vardefunc.mask import FDOG, ExLaplacian4, MinMax, SobelStd
from vsutil import depth, iterate

core = vs.core



class Thr(NamedTuple):
    lo: float
    hi: float


class Denoise:
    @staticmethod
    def bm3d(clip: vs.VideoNode, sigma: List[float], radius: int, **kwargs: Any) -> vs.VideoNode:
        return cast(vs.VideoNode, eoe.denoise.BM3D(clip, sigma, radius=radius, CUDA=True, **kwargs))


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
