"""Mushoku script"""
__author__ = 'Vardë'


from functools import partial
from typing import cast

import awsmfunc as awf
import havsfunc as hvf
import mvsfunc as mvf
import vapoursynth as vs
from G41Fun import MaskedDHA
from lvsfunc.util import replace_ranges as rfs
from vardautomation import FileInfo, PresetAAC, PresetBD, PresetChapXML
from vardefunc.deband import dumb3kdb
from vardefunc.mask import (FDOG, ExLaplacian4, PrewittStd, detail_mask,
                            region_mask)
from vardefunc.misc import merge_chroma
from vardefunc.noise import decsiz
from vsutil import depth, get_y, join, plane, split

from mushoku_common import Encoding, fake_rescale

core = vs.core



JPBD = FileInfo(
    r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00004.m2ts', 24, -24,
    preset=[PresetBD, PresetAAC, PresetChapXML]
)
JPBD.do_qpfile = True
PART1, PART2, EDSTART, PREVIEW = 4245, 12421, 31768, 33926
OPEND, EDEND = PART1 - 1, PREVIEW - 1


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


        fixedge_a = awf.bbmod(out, 1, 1, 1, 1, 20, blur=700, u=False, v=False)

        fixedge = out
        fixedge = rfs(fixedge, fixedge_a, [(EDSTART + 309, EDEND)])
        out = fixedge
        out = depth(out, 16)



        dehalo = MaskedDHA(out, rx=1.4, ry=1.4, darkstr=0.02, brightstr=1)
        dehalo = rfs(out, dehalo, [(EDEND + 1, src.num_frames - 1)])
        out = dehalo



        """IRL PART"""
        luma = get_y(out)
        lineart = FDOG().get_mask(luma)
        fkrescale = fake_rescale(luma, 844, b=0, c=0.5, coef_dering=1.4)
        masked = core.std.MaskedMerge(luma, fkrescale, lineart)

        merged = merge_chroma(masked, out)
        merged = rfs(out, merged, [(24712, 24843)])
        out = merged
        """"""


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
