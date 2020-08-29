"""Symphogear G script"""
__author__ = 'Vardë'

from typing import Optional, Dict, Any

import vardefunc as vdf
import mvsfunc as mvf

from vsutil import get_y
import vapoursynth as vs

core = vs.core


def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                   knlm_args: Optional[Dict[str, Any]] = None,
                   bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
    """Denoise luma with BM3D and chroma with knlmeansCL

    Args:
        clip (vs.VideoNode): Source clip.
        knlm_h (float, optional): h parameter in knlm.KNLMeansCL. Defaults to 0.5.
        sigma (float, optional): Sigma parameter in mvf.BM3D. Defaults to 2.
        knlm_args (Optional[Dict[str, Any]], optional): Optional extra arguments for knlm.KNLMeansCL. Defaults to None.
        bm3d_args (Optional[Dict[str, Any]], optional): Optional extra arguments for mvf.BM3D. Defaults to None.

    Returns:
        vs.VideoNode: [description]
    """
    knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
    if knlm_args is not None:
        knargs.update(knlm_args)

    b3args = dict(radius1=1, profile1='fast')
    if bm3d_args is not None:
        b3args.update(bm3d_args)

    luma = get_y(clip)
    luma = mvf.BM3D(luma, sigma, **b3args)
    chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

    return vdf.merge_chroma(luma, chroma)



def single_rate_antialiasing(clip: vs.VideoNode, rep: Optional[int] = None,
                             **eedi3_args: Any)-> vs.VideoNode:
    """Drop half the field with eedi3+nnedi3 and interpolate them.

    Args:
        clip (vs.VideoNode): Source clip.
        rep (Optional[int], optional): Repair mode. Defaults to None.

    Returns:
        vs.VideoNode: [description]
    """
    nnargs: Dict[str, Any] = dict(nsize=0, nns=3, qual=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    eeargs.update(eedi3_args)

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3, core.nnedi3cl.NNEDI3CL

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgsf.Repair(flt, clip, rep)

    return flt
