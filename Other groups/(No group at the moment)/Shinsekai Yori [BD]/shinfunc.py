from typing import Callable, Tuple, List, Union
from functools import partial

import mvsfunc as mvf
import G41Fun as gf
import debandshit as dbs
import vardefunc as vdf
import placebo


from vsutil import depth, get_y, get_w, iterate
import vapoursynth as vs

core = vs.core


# Upscale functions
def nnedi3_upscale(clip: vs.VideoNode, factor: float, args: dict)-> vs.VideoNode:
    h = clip.height*factor
    w = get_w(h)
    clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **args).std.Transpose().nnedi3.nnedi3(0, True, **args)
    return core.resize.Spline36(clip, w, h, src_top=.5, src_left=.5)

def eedi3_upscale(clip: vs.VideoNode, factor: float, eeargs: dict, nnargs: dict)-> vs.VideoNode:
    h = clip.height*factor
    w = get_w(h)
    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)
    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)
    return core.resize.Spline36(clip, w, h, src_top=.5, src_left=.5)

def sangnom_upscale(clip: vs.VideoNode, factor: float, aa: int)-> vs.VideoNode:
    h = clip.height*factor
    w = get_w(h)
    upscale = clip.std.Transpose().sangnom.SangNom(2, True, aa).std.Transpose().sangnom.SangNom(2, True, aa)
    return core.resize.Spline36(upscale, w, h, src_top=.5, src_left=.5)

def fsrcnnx_upscale(clip: vs.VideoNode, height: int, shader_file: str,
                    downscaler: Callable[[vs.VideoNode, int, int], vs.VideoNode])-> vs.VideoNode:
    clip = depth(clip, 16)

    fsrcnnx = placebo.shader(clip, clip.width*2, clip.height*2, shader_file)
    smooth = nnedi3_upscale(clip, 2, dict(nsize=4, nns=4, qual=2, pscrn=2))

    noringing = core.std.Expr([fsrcnnx, smooth], 'x y < x y ?')

    downscale = downscaler(noringing, get_w(height, clip.width/clip.height), height)

    return depth(downscale, 32)



# Various masks
def perform_credit_mask(clip: vs.VideoNode, thr: Union[int, float], **args)-> vs.VideoNode:
    mask = vdf.drm(clip, **args).std.Binarize(thr)
    mask = iterate(mask, core.std.Inflate, 4)
    mask = iterate(mask, core.std.Deflate, 2)
    return mask

def detail_mask(clip: vs.VideoNode, brz_a: Tuple[int, int], brz_b: Tuple[int, int],
                rad: List[Tuple[int, int]], **ret_args)-> vs.VideoNode:
    luma = get_y(clip)

    ret = core.retinex.MSRCP(luma, **ret_args)

    mask_a = dbs.rangemask(ret, rad[0][0], rad[0][1]).std.Binarize(brz_a[0])
    mask_b = gf.EdgeDetect(ret, 'FDOG').std.Binarize(brz_b[0])
    dark = core.std.Expr([mask_a, mask_b], 'x y max')


    mask_a = dbs.rangemask(luma, rad[1][0], rad[1][1]).std.Binarize(brz_a[1])
    mask_b = gf.EdgeDetect(luma, 'FDOG').std.Binarize(brz_b[1])
    light = core.std.Expr([mask_a, mask_b], 'x y max')


    return core.std.Expr([dark, light], 'x y max').rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)

def edge_mask_simple(clip: vs.VideoNode, mode: str, thr: int, mpand: Tuple[int, int])-> vs.VideoNode:
    coord = [1, 2, 1, 2, 2, 1, 2, 1]
    mask = gf.EdgeDetect(clip, mode).std.Binarize(thr)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=coord), mpand[0])
    mask = iterate(mask, partial(core.std.Minimum, coordinates=coord), mpand[1])
    return mask.std.Inflate().std.Deflate()



# Deband function
def deband_stonks(clip: vs.VideoNode, radius: int, thr: int, iterations: int, mask: vs.VideoNode)-> vs.VideoNode:
    deband = placebo.deband(clip, radius, thr, iterations)
    deband = mvf.LimitFilter(deband, clip, thr=0.6, elast=3.0)
    return core.std.MaskedMerge(deband, clip, mask)
