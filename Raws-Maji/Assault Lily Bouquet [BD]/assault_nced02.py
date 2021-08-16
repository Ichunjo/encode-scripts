"""Assault Lily Bouquet script"""
__author__ = 'Vardë'

from typing import Tuple

import vapoursynth as vs
import vardefunc as vdf
from vardautomation import FileInfo, PresetAAC, PresetBD, PresetChapXML
from vardefunc.mask import MinMax, SobelStd
from vardefunc.misc import merge_chroma
from vardefunc.noise import Graigasm
from vsutil import depth, get_y, iterate

from assault_common import (AA, Denoise, Encoding, Mask, Scale, SetFrameProp,
                            Thr, graigasm_args)

core = vs.core


JPBD = FileInfo('[BDMV][210224][BRMM-10342][アサルトリリィ BOUQUET 2]/BRMM_10342/BDMV/STREAM/00003.m2ts',
                (None, 2085), preset=(PresetBD, PresetAAC, PresetChapXML))
JPBD.do_qpfile = True

CROP_BORDERS_ED = True


class Filtering:
    def main(self) -> vs.VideoNode:
        src = JPBD.clip_cut
        # while src.num_frames < 2160:
        #     src += src[-1]
        src = SetFrameProp.yuv_hd(src)
        src = depth(src, 16)
        out = src


        # Limited aliasing + contra
        y = get_y(out)
        lmask = SobelStd().get_mask(y, lthr=75 << 8, hthr=75 << 8).std.Maximum().std.Deflate()
        # -------------------------- SCENEFILTERING AA --------------------------
        aaa = AA.upscaled_sraa(y)
        # -------------------------- SCENEFILTERING AA --------------------------
        aaa = core.std.MaskedMerge(y, aaa, lmask)
        out = merge_chroma(aaa, out)


        # Denoising using BM3D CUDA
        denoise = Denoise.bm3d(depth(out, 32), [1.5, 2.5, 2.5], radius=1, profile='fast')
        out = denoise


        # Convert to RGB and deband the shit out of it
        # Blue plane seems to have more banding
        rgb = SetFrameProp.rgb(Scale.to_444(out).resize.Point(format=vs.RGBS))
        y = get_y(out)
        y_db = vdf.placebo.deband(y, 20, threshold=6.0, iterations=1, grain=2.0)
        rgb_db_a = vdf.placebo.deband(rgb, 20, threshold=[5.5, 7.0, 7.5], iterations=1, grain=2.0)
        rgb_db_b = vdf.placebo.deband(rgb, 24, threshold=[6.25, 7.5, 10.1], iterations=3, grain=2.5)

        # Build masks
        lmask, mmmask = self.deband_masks(y)
        # Merge
        deband_rgb = core.std.MaskedMerge(rgb_db_b, rgb_db_a, mmmask)
        deband_rgb = core.std.MaskedMerge(deband_rgb, rgb, lmask)

        y_db = core.std.MaskedMerge(y_db, y, lmask)
        deband = SetFrameProp.yuv_hd(
            merge_chroma(luma=y_db, ref=Scale.to_yuv420(deband_rgb, out))
        )
        out = depth(deband, 16)


        # My graining nonsense
        pref = iterate(get_y(out), core.std.Maximum, 2).std.Convolution([1]*9)
        grain = Graigasm(**graigasm_args).graining(out, prefilter=pref)  # type: ignore
        grain = core.std.MaskedMerge(grain, out, Mask.limit_grain_mask(src))
        out = grain


        # Fix letterboxes ending
        if CROP_BORDERS_ED:
            crop = core.std.Crop(out, 0, 0, 132, 132).edgefixer.ContinuityFixer(0, 1, 0, 0)
            crop = core.std.AddBorders(crop, 0, 0, 132, 132)
            out = crop

        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


    @staticmethod
    def deband_masks(y: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
        slocation = [(0.0, 0.5), (0.10, 2.0), (0.2, 10.0), (0.5, 25.0), (1.0, 1.5)]
        pre = core.dfttest.DFTTest(y, slocation=sum(slocation, ()))
        lmask = Mask().lineart_deband_mask(
            pre, brz_rg=2100/65536, brz_ed=1500/65536, brz_ed_ret=9750/65536,
            ret_thrs=Thr(lo=(19 - 16) / 219, hi=(22.5 - 16) / 219)
        )
        mmmask = MinMax(18, 0).get_mask(y).std.Binarize(2600/65536).std.BoxBlur(0, 5, 10, 5, 10)
        return lmask, mmmask


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = Encoding(JPBD, filtered)
    brrrr.run(merge_chapters=False)
    brrrr.cleanup()
else:
    JPBD.clip_cut.set_output(0)
    # FILTERED = Filtering().main()
    # if not isinstance(FILTERED, vs.VideoNode):
    #     for i, clip_filtered in enumerate(FILTERED, start=1):
    #         clip_filtered.set_output(i)
    # else:
    #     FILTERED.set_output(3)
