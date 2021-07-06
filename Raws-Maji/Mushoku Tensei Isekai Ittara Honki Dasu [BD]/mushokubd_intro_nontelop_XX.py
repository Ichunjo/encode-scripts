"""Mushoku script"""
__author__ = 'Vardë'


from functools import partial
from typing import cast

import havsfunc as hvf
import mvsfunc as mvf
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetBD, VPath
from vardefunc.deband import dumb3kdb
from vardefunc.mask import ExLaplacian4, PrewittStd, detail_mask, region_mask
from vardefunc.misc import merge_chroma
from vardefunc.noise import decsiz
from vsutil import depth, get_y, join, plane, split

from mushoku_common import Encoding

core = vs.core



JPBDS = [
    FileInfo(
        r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00005.m2ts', 24, -3,
        preset=[PresetBD, PresetAAC]
    ),
    FileInfo(
        r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00006.m2ts', 24, None,
        preset=[PresetBD, PresetAAC]
    ),
    FileInfo(
        r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00007.m2ts', 24, None,
        preset=[PresetBD, PresetAAC]
    ),
    FileInfo(
        r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00008.m2ts', 24, None,
        preset=[PresetBD, PresetAAC]
    )
]


class Filtering:
    def main(self, file: FileInfo) -> vs.VideoNode:
        """Vapoursynth filtering"""
        src = file.clip_cut
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
    for i, JPBD in enumerate(JPBDS):
        JPBD.name = f'mushokubd_intro_nontelop_{i+2:02.0f}'
        JPBD.name_clip_output = VPath(JPBD.name + '.265')
        JPBD.name_file_final = VPath(JPBD.name + '.mkv')
        filtered = Filtering().main(JPBD)
        brrrr = Encoding(JPBD, filtered)
        if not JPBD.name_file_final.exists():
            brrrr.run(do_chaptering=False)
            brrrr.cleanup()
else:
    JPBDS[1].clip_cut.set_output(0)
    FILTERED = Filtering().main(JPBDS[1])
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)
