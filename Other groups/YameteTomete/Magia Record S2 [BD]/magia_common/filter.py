from typing import Any, List, NamedTuple, Optional, Sequence, Set, Tuple, cast

import cv2
import numpy as np
import vapoursynth as vs
from havsfunc import LSFmod
from lvsfunc.kernels import Bicubic
from lvsfunc.types import Range
from lvsfunc.util import replace_ranges, scale_thresh
from vardefunc.aa import Eedi3SR, Nnedi3SS, upscaled_sraa
from vardefunc.mask import FDOG, Difference, ExLaplacian4, MinMax, SobelStd
from vardefunc.scale import to_444
from vardefunc.util import get_sample_type, select_frames
from vsutil import Dither, depth, get_depth, get_y, insert_clip, iterate
from vsutil.info import scale_value

core = vs.core



class Thr(NamedTuple):
    lo: float
    hi: float


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


def insert_frame(clip: vs.VideoNode, frame_num: int, number: int) -> vs.VideoNode:
    indices = list(range(frame_num)) + [frame_num] * number + list(range(frame_num, clip.num_frames))
    return select_frames(clip, indices)
