from typing import Any, List, NamedTuple, cast

import EoEfunc as eoe
import vapoursynth as vs
from vardefunc.mask import FDOG, ExLaplacian4, MinMax, SobelStd
from vardefunc.scale import nnedi3_upscale
from vsutil import depth, get_y, iterate

core = vs.core


class Denoise:
    @staticmethod
    def bm3d(clip: vs.VideoNode, sigma: List[float], radius: int, **kwargs: Any) -> vs.VideoNode:
        return cast(vs.VideoNode, eoe.denoise.BM3D(clip, sigma, radius=radius, CUDA=True, **kwargs))


class Scale:
    @staticmethod
    def waifu2x(clip: vs.VideoNode, noise: int = 1) -> vs.VideoNode:
        clip = get_y(depth(clip, 32))

        lineart = SobelStd().get_mask(clip).std.Maximum().std.Minimum().resize.Bilinear(clip.width * 2, clip.height * 2)

        nnedi3 = nnedi3_upscale(clip)
        w2x = core.w2xnvk.Waifu2x(
            core.resize.Point(clip, format=vs.RGBS), noise=noise, scale=2, model=0
        ).resize.Point(format=vs.GRAYS, matrix=1, dither_type='error_diffusion')

        upscale = core.std.MaskedMerge(nnedi3, w2x, lineart)

        diff = core.std.Expr(
            (
                upscale.resize.Bilinear(
                    clip.width, clip.height
                ).std.BoxBlur(0, 1, 1, 1, 1).resize.Bilinear(clip.width * 2, clip.height * 2),
                nnedi3.resize.Bilinear(
                    clip.width, clip.height
                ).std.BoxBlur(0, 1, 1, 1, 1).resize.Bilinear(clip.width * 2, clip.height * 2)
            ),
            'x y - abs'
        )

        th_lo, th_hi = 3500 / 65535, 7000 / 65535
        strength = f'{th_hi} x - {th_hi} {th_lo} - /'
        upscale = core.std.Expr(
            (diff, nnedi3, upscale),
            f'x {th_lo} < z x {th_hi} > y z ' + strength + ' * y 1 ' + strength + ' - * + ? ?'
        ).rgsf.Repair(nnedi3, 13).rgsf.Repair(nnedi3, 7)

        return upscale
