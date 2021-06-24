# noqa
from typing import Optional, Dict, Any

import vardefunc as vdf
from vsutil import get_w

import vapoursynth as vs
core = vs.core


class AA():
    def upscaled_sraaaaaaaaaa(self, clip: vs.VideoNode, height: int) -> vs.VideoNode:
        bicu_args = dict(filter_param_a=-0.5, filter_param_b=0.25)
        ssing = vdf.scale.nnedi3_upscale(clip) \
            .resize.Bicubic(get_w(height), height, **bicu_args)

        sraing = self.sraa_eedi3(ssing, 9, alpha=0.2, beta=0.6, gamma=40, mdis=15, nrad=2)

        return core.resize.Bicubic(sraing, clip.width, clip.height, **bicu_args)

    @staticmethod
    def sraa_eedi3(clip: vs.VideoNode, rep: Optional[int] = None, **eedi3_args: Any) -> vs.VideoNode:
        """Drop half the field with eedi3+nnedi3 and interpolate them.

        Args:
            clip (vs.VideoNode): Source clip.
            rep (Optional[int], optional): Repair mode. Defaults to None.

        Returns:
            vs.VideoNode: AA'd clip
        """
        nnargs: Dict[str, Any] = dict(nsize=0, nns=3, qual=1)
        eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
        eeargs.update(eedi3_args)

        eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3CL, core.nnedi3.nnedi3

        flt = core.std.Transpose(clip)
        flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
        flt = core.std.Transpose(flt)
        flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

        if rep:
            flt = core.rgvs.Repair(flt, clip, rep)

        return flt

    @staticmethod
    def sraa_sangnom(clip: vs.VideoNode, rep: Optional[int] = None, aa: int = 48) -> vs.VideoNode:
        flt = core.std.Transpose(clip)
        flt = core.sangnom.SangNom(flt, 1, False, aa=aa)
        flt = core.std.Transpose(flt)
        flt = core.sangnom.SangNom(flt, 1, False, aa=aa)

        if rep:
            flt = core.rgvs.Repair(flt, clip, rep)

        return flt

    def upscaled_sraa_sangnom(self, clip: vs.VideoNode, height: int, aa: int = 48) -> vs.VideoNode:
        bicu_args = dict(filter_param_a=-0.5, filter_param_b=0.25)
        ssing = vdf.scale.eedi3_upscale(clip).resize.Bicubic(get_w(height), height, **bicu_args)

        sraeedi3ing = self.sraa_eedi3(ssing, None, alpha=0.8, beta=0.2, gamma=20, mdis=25, nrad=3)
        sraesgning = self.sraa_sangnom(sraeedi3ing, 9, aa)

        return core.resize.Bicubic(sraesgning, clip.width, clip.height, **bicu_args)
