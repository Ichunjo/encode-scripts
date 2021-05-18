
"""Love Live Script script"""
from __future__ import annotations

__author__ = 'Vardë'

from functools import partial
from typing import Any, Dict, Optional

import G41Fun as gf
import havsfunc as hvf
import lvsfunc as lvf
import muvsfunc as muvf
import mvsfunc as mvf
import vardefunc as vdf
from vardautomation import FileInfo, PresetBD, PresetFLAC
from vsutil import depth, get_w, get_y, join

import vapoursynth as vs

from love_live_common import EncodeGoBrrr, FileInfoMore


core = vs.core

JPBD = FileInfoMore(
    r'[BDMV][210127][BCXA-1591][ラブライブ！虹ヶ咲学園スクールアイドル同好会][第2巻]\BDROM\BDMV\STREAM\00005.m2ts', 24, -25,
    preset=[PresetBD, PresetFLAC]
)
JPBD_NCOP = FileInfo(
    r'[BDMV][210127][BCXA-1591][ラブライブ！虹ヶ咲学園スクールアイドル同好会][第2巻]\BDROM\BDMV\STREAM\00010.m2ts', 24, -24,
)


class Filtering():  # noqa
    def main(self: Filtering) -> vs.VideoNode:
        """Vapoursynth filtering"""
        src = JPBD.clip_cut
        src = depth(src, 16)
        out = src


        h = 800  # noqa
        w = get_w(h)  # noqa
        opstart, opend = 3837, 5993
        edstart, edend = 31170, 33326



        inp = get_y(out)
        out = inp



        # Remove the grain
        ref = hvf.SMDegrain(out, tr=1, thSAD=300, plane=0)
        preden = mvf.BM3D(out, sigma=2, radius1=1, ref=ref)
        out = preden




        # Rescale / Antialiasing / Limiting
        out = depth(out, 32)
        lineart = vdf.mask.FDOG().get_mask(out, lthr=0.065, hthr=0.065).std.Maximum().std.Minimum()
        lineart = lineart.std.Median().std.Convolution([1] * 9)


        descale_clips = [core.resize.Bicubic(out, w, h, filter_param_a=1/3, filter_param_b=1/3),
                         core.descale.Debicubic(out, w, h, 0, 1 / 2),
                         core.descale.Debilinear(out, w, h)]
        descale = core.std.Expr(descale_clips, 'x y z min max y z max min z min')

        upscale = vdf.scale.fsrcnnx_upscale(descale, height=h * 2, shader_file=r'_shaders\FSRCNNX_x2_56-16-4-1.glsl',
                                            upscaled_smooth=vdf.scale.eedi3_upscale(descale), profile='zastin',
                                            sharpener=partial(gf.DetailSharpen, sstr=1.65, power=4, mode=0, med=True))


        antialias = self.sraa_eedi3(upscale, 3, alpha=0.2, beta=0.4, gamma=100, mdis=20, nrad=3)

        downscale = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic', filter_param_a=0, filter_param_b=0)

        adaptmask = core.adg.Mask(downscale.std.PlaneStats(), 25).std.Minimum().std.Minimum().std.Convolution([1] * 9)
        contra = gf.ContraSharpening(downscale, depth(preden, 32), radius=2).rgsf.Repair(downscale, 1)
        contra = core.std.MaskedMerge(downscale, contra, adaptmask)


        scaled = core.std.MaskedMerge(out, contra, lineart)
        merged = vdf.misc.merge_chroma(depth(scaled, 16), src)
        out = merged


        detail_light_mask = lvf.mask.detail_mask(out, brz_a=1500, brz_b=600)

        deband = vdf.deband.dumb3kdb(out, 16, [33, 1], sample_mode=4, use_neo=True)
        deband = core.std.MaskedMerge(deband, out, detail_light_mask)
        out = deband



        # Restore the grain
        neutral = inp.std.BlankClip(960, 540, color=128 << 8)
        diff = join([inp.std.MakeDiff(preden), neutral, neutral])
        grain = core.std.MergeDiff(out, diff)
        out = grain



        ref = src
        creditless_mask = vdf.mask.diff_creditless_mask(
            ref, src[opstart:opend+1], JPBD_NCOP.clip_cut[:opend-opstart+1], opstart, thr=25 << 8, sw=3, sh=3, prefilter=True
        ).std.Deflate()
        ringing_mask = hvf.HQDeringmod(ref, mrad=1, msmooth=2, mthr=40, show=True)

        credit = out
        credit = lvf.rfs(credit, ref, [(edstart, edend)])
        credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, creditless_mask, 0), [(opstart, opend)])
        credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, ringing_mask, 0),
                         [(opstart + 169, opstart + 411), (29091, 29116), (29309, 29334)])
        out = credit



        endcard = out + out[31476] * 119
        out = endcard


        decs = vdf.noise.decsiz(out, sigmaS=10, min_in=110 << 8, max_in=192 << 8, gamma=1.1)
        out = decs


        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2])


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

        eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3CL, core.nnedi3cl.NNEDI3CL

        flt = core.std.Transpose(clip)
        flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
        flt = core.std.Transpose(flt)
        flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

        if rep:
            flt = core.rgsf.Repair(flt, clip, rep)

        return flt


if __name__ == '__main__':
    wizardry = EncodeGoBrrr(Filtering().main(), JPBD)
    wizardry.run()
    wizardry.cleanup()
else:
    JPBD.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    FILTERED.set_output(1)
    # FILTERED[0].set_output(1)
    # FILTERED[1].set_output(2)
