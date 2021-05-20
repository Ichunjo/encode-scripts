"""YuYuYu Churutto script"""  # noqa
from __future__ import annotations

__author__ = 'Vardë'

import G41Fun as gf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vdf
import xvs
from vardautomation import FileInfo, PresetEAC3, PresetWEB
from vsutil import depth, get_y

from churutto_common import EncodeGoBrrr

import vapoursynth as vs
core = vs.core



NUM = __file__[-5:-3]
WEB = FileInfo(f'{NUM}/Yuuki Yuuna wa Yuusha de Aru Churutto! - {NUM} (Amazon Rental VBR 1080p).mkv', 24, -23,
               preset=[PresetWEB, PresetEAC3])


class Filtering():  # noqa
    def main(self: Filtering) -> vs.VideoNode:  # noqa
        """Vapoursynth filtering"""
        src = WEB.clip_cut
        src = src.std.AssumeFPS(src)


        src = depth(src, 16)
        out = src


        denoise = hvf.SMDegrain(out, tr=1, thSAD=100, thSADC=100)
        out = denoise


        dering = hvf.EdgeCleaner(out, 15, smode=1, hot=True)
        dering = gf.MaskedDHA(dering, darkstr=0.05, brightstr=0.75)
        out = dering


        aaa = vdf.scale.nnedi3_upscale(get_y(out), pscrn=1, correct_shift=False)
        aaa = aaa.resize.Bicubic(1920, 1080, src_left=0.5, src_top=0.5, filter_param_a=-0.5, filter_param_b=0.25)
        out = vdf.misc.merge_chroma(aaa, out)


        cwarp = xvs.WarpFixChromaBlend(out, 64, depth=4)
        out = cwarp


        detail_mask = lvf.mask.detail_mask(out, brz_a=2250, brz_b=1200)
        deband = vdf.deband.dumb3kdb(out, 16, threshold=33, grain=[24, 0])
        deband = core.std.MaskedMerge(deband, out, detail_mask)
        out = deband


        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])[:100]



if __name__ == '__main__':
    wizardry = EncodeGoBrrr(Filtering().main(), WEB)
    wizardry.run()
else:
    WEB.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    FILTERED.set_output(1)
    # FILTERED[0].set_output(1)
    # FILTERED[1].set_output(2)
