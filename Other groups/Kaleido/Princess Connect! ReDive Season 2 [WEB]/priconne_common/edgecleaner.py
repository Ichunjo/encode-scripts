from __future__ import annotations

from math import ceil, floor

import vapoursynth as vs
# https://github.com/Varde-s-Forks/RgToolsVS
from rgvs import removegrain, repair
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, PrewittStd
from vsmask.types import ensure_format as _ensure_format
from vsutil import Dither
from vsutil import Range as CRange
from vsutil import depth, scale_value

core = vs.core

# EdgeCleaner() v1.04 (03/04/2012)
# - a simple edge cleaning and weak dehaloing function
#
# Requirements:
# AWarpSharp2, RGVS (optional)
#
# Parameters:
# strength (float)      - specifies edge denoising strength (10)
# rep (boolean)         - actives Repair for the aWarpSharped clip (true)
# rmode (integer)       - specifies the Repair mode;
#                         1 is very mild and good for halos,
#                         16 and 18 are good for edge structure preserval on strong settings but keep more halos and edge noise,
#                         17 is similar to 16 but keeps much less haloing, other modes are not recommended (17)
# smode (integer)       - specifies what method will be used for finding small particles, ie stars;
#                         0 is disabled, 1 uses RemoveGrain (0)
# hot (boolean)         - specifies whether removal of hot pixels should take place (false)


def edge_cleaner(clip: vs.VideoNode, strength: float = 10, rmode: int = 17, hot: bool = False, smode: int = 0, edgemask: EdgeDetect = PrewittStd()) -> vs.VideoNode:
    """
    Simple edge cleaning and weak dehaloing function

    :param clip:            Source clip
    :param strength:        Edge denoising strength
    :param rmode:           Repair mode:
                            * 1 is very mild and good for halos
                            * 16 and 18 are good for edge structure preserval on strong settings but keep more halos and edge noise,
                            * 17 is similar to 16 but keeps much less haloing, other modes are not recommended
    :param hot:             Remove hot pixels
    :param smode:           Find small particles, eg. stars
    :param edgemask:        Internal mask used for detecting the edges
    :return:                Edge cleaned clip
    """
    clip = _ensure_format(clip)
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo: format not supported')

    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1 if clip.format.sample_type == vs.INTEGER else 1.0

    clip_y, *chroma = split(clip)

    if smode > 0:
        strength += 4

    main = padding(clip_y, 6, 6, 6, 6)
    # warpsf is way too slow
    main = depth(main, 16, vs.INTEGER, dither_type=Dither.NONE) if clip.format.sample_type == vs.FLOAT else main
    main = main.warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    main = depth(main, bits, clip.format.sample_type, dither_type=Dither.NONE)
    #
    main = repair(main, clip_y, rmode)

    mask = edgemask.get_mask(clip_y).std.Expr(
        f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(32, 8, bits, CRange.FULL)} > {peak} x ? ?'
    ).std.InvertMask().std.Convolution([1] * 9)

    final = core.std.MaskedMerge(clip_y, main, mask)

    if hot:
        final = repair(final, clip_y, 2)
    if smode:
        clean = removegrain(clip_y, 17)
        diff = core.std.MakeDiff(clip_y, clean)
        expr = f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(16, 8, bits, CRange.FULL)} > {peak} x ? ?'
        mask = edgemask.get_mask(
            diff.std.Levels(scale_value(40, 8, bits, CRange.FULL), scale_value(168, 8, bits, CRange.FULL), 0.35)
        )
        mask = removegrain(mask, 7).std.Expr(expr)
        final = core.std.MaskedMerge(final, clip_y, mask)

    return join([final] + chroma, clip.format.color_family)


def padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    return clip.resize.Point(
        clip.width + left + right, clip.height + top + bottom,
        src_left=-left, src_top=-top,
        src_width=clip.width + left + right, src_height=clip.height + top + bottom
    )


def cround(x: float) -> int:
    return floor(x + 0.5) if x > 0 else ceil(x - 0.5)


def mod_x(x: int, val: int | float) -> int:
    return max(x * x, cround(val / x) * x)
