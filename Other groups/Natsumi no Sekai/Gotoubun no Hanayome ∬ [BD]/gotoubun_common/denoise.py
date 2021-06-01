# noqa
from typing import Optional, Dict, Any

import havsfunc as hvf
import mvsfunc as mvf
import vardefunc as vdf
from vsutil import get_y

import vapoursynth as vs
core = vs.core


class Denoise:
    @staticmethod
    def ref_denoise(clip: vs.VideoNode, **kwargs) -> vs.VideoNode:
        return hvf.SMDegrain(clip, **kwargs)

    @staticmethod
    def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                       knlm_args: Optional[Dict[str, Any]] = None, bm3d_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
        knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
        if knlm_args is not None:
            knargs.update(knlm_args)

        b3args = dict(radius1=1, profile1='fast')
        if bm3d_args is not None:
            b3args.update(bm3d_args)

        luma = get_y(clip)
        luma = mvf.BM3D(luma, sigma, **b3args)
        chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

        return vdf.misc.merge_chroma(luma, chroma)
