"""Mushoku script"""
__author__ = 'Vardë'


from functools import partial
from typing import cast

import havsfunc as hvf
from lvsfunc.util import replace_ranges as rfs
import mvsfunc as mvf
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetBD, PresetChapXML
from vardefunc.deband import dumb3kdb
from vardefunc.mask import ExLaplacian4, FDOG, PrewittStd, detail_mask, region_mask
from vardefunc.misc import merge_chroma
from vardefunc.noise import decsiz
from vsutil import depth, get_y, join, plane, split

from mushoku_common import Encoding, fake_rescale

core = vs.core



# from vardautomation import UNDEFINED, MatroskaXMLChapters, MplsReader
# reader = MplsReader(r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM', UNDEFINED)
# reader.write_playlist('chapters', chapters_obj=MatroskaXMLChapters)
# exit()

JPBD = FileInfo(
    r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00000.m2ts', 24, -24,
    preset=[PresetBD, PresetAAC, PresetChapXML]
)


CREDITS = [
    (0, 131), (204, 267), (337, 400), (611, 682), (800, 864),
    (1233, 1298), (2013, 2259), (31769, 33926)
]



class Filtering:
    def main(self) -> vs.VideoNode:
        """Vapoursynth filtering"""
        src = JPBD.clip_cut
        out = src



        luma = get_y(out)
        rows = [core.std.CropAbs(luma, out.width, 1, top=out.height - 1),
                core.std.CropAbs(luma, out.width, 1, top=out.height - 2)]
        diff = core.std.Expr(rows, 'x y - abs').std.PlaneStats()

        row_fix = merge_chroma(
            luma.fb.FillBorders(bottom=1, mode="fillmargins"),
            out.fb.FillBorders(bottom=2, mode="fillmargins")
        )

        fixrow = core.std.FrameEval(out, partial(self._select_row, clip=out, row_fix=row_fix), prop_src=diff)
        out = fixrow
        out = depth(out, 16)





        luma = get_y(out)
        # return upscaled_sraa(luma, 1.5, singlerater=Eedi3SR(eedi3cl=True, nnedi3cl=True, alpha=0.2, beta=0.5, gamma=400))
        lineart = FDOG().get_mask(luma)
        fkrescale = fake_rescale(luma, 844, b=0, c=0.5, coef_dering=1.4)
        masked = core.std.MaskedMerge(luma, fkrescale, lineart)

        merged = merge_chroma(masked, out)
        merged = rfs(out, merged, [(0, 454), (718, 980), (1299, 2012)])
        out = merged





        # Denoising only the chroma
        pre = hvf.SMDegrain(out, tr=2, thSADC=300, plane=3)
        planes = split(out)
        planes[1], planes[2] = [mvf.BM3D(planes[i], 1.25, radius2=2, pre=plane(pre, i)) for i in range(1, 3)]
        out = join(planes)


        preden = core.dfttest.DFTTest(out, sbsize=16, sosize=12, tbsize=1)
        detailmask = core.std.Expr(
            split(
                detail_mask(preden, brz_mm=2500, brz_ed=1400, edgedetect=self.ExLaplaWitt()).resize.Bilinear(format=vs.YUV444P16)
            ), 'x y z max max'
        )

        deband = dumb3kdb(preden, 16, 30, grain=[24, 0])
        deband = core.std.MergeDiff(deband, out.std.MakeDiff(preden))
        deband = core.std.MaskedMerge(deband, out, detailmask)
        out = deband



        decz = decsiz(out, min_in=128 << 8, max_in=192 << 8)
        out = decz



        ref = depth(src, 16)
        credit = out
        credit = rfs(out, ref, CREDITS)
        out = credit



        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

    @staticmethod
    def _select_row(n, f: vs.VideoFrame, clip: vs.VideoNode, row_fix: vs.VideoNode, thr: float = 1e-2) -> vs.VideoNode:
        return row_fix if cast(float, f.props['PlaneStatsAverage']) > thr else clip

    class ExLaplaWitt(ExLaplacian4):
        def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
            exlaplacian4 = super()._compute_mask(clip)
            prewitt = PrewittStd().get_mask(clip)
            mask = core.std.Expr((exlaplacian4, prewitt), 'x y max')
            return region_mask(mask, right=2).fb.FillBorders(right=2)


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = Encoding(JPBD, filtered)
    brrrr.chaptering(0-JPBD.frame_start)  # type: ignore
    brrrr.run()
    brrrr.cleanup()
else:
    JPBD.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)
