from typing import Any, List, cast

import EoEfunc as eoe
import fvsfunc
import lvsfunc
import vapoursynth as vs
import vardefunc as vdf
import xvs
from G41Fun import MaskedDHA
from vardautomation import FileInfo, PresetAAC, PresetGeneric, X265Encoder
from vardautomation.patch import Patch
from vardautomation.tooling import EztrimCutter, Mux
from vardefunc.mask import FDOG, SobelStd, detail_mask
from vardefunc.misc import merge_chroma
from vardefunc.scale import nnedi3_upscale, to_444
from vsutil import depth, get_y, iterate

core = vs.core


NUM = __file__[-5:-3]

TV_NHK = FileInfo(
    f'{NUM}/Love Live! Superstar!! - {NUM} (NHKE).dgi',
    (306, None),
    idx=core.dgdecodenv.DGSource,
    preset=(PresetGeneric, PresetAAC)
)


class Filtering:
    def main(self) -> vs.VideoNode:
        src = TV_NHK.clip_cut
        # src_c = src
        src = depth(src, 16)

        ivtc = fvsfunc.JIVTC(src, 1)
        deinterlace = core.nnedi3.nnedi3(ivtc, 0)
        combed_frames = [19255, 1990, 34046, 18438, 20556, 20756, 21233, 21657, 21824, 22180, 33926]
        ivtc = lvsfunc.util.replace_ranges(ivtc, deinterlace, combed_frames)  # type: ignore
        shidt_combed_frames = list(map(lambda x: x-1, combed_frames))
        ivtc = core.std.FreezeFrames(ivtc, shidt_combed_frames, combed_frames, shidt_combed_frames)
        out = ivtc

        deblock = fvsfunc.AutoDeblock(out)
        out = deblock


        full = self.to_444(out)
        out = full


        chromafix = core.fb.FillBorders(out, 3, 3)
        lumafix = core.edgefixer.ContinuityFixer(out, [2, 0, 0], 0, [2, 0, 0], 0, [2, 0, 0])
        out = merge_chroma(lumafix, chromafix)



        denoise = self.bm3d(out, [1.5, 1.0, 1.0], 1)
        lineart = SobelStd().get_mask(get_y(denoise), lthr=10000, multi=2.5)
        lineart = iterate(lineart, core.std.Maximum, 6).std.Convolution([1]*9)

        hard_den = self.bm3d(denoise, [10, 7.5, 7.5], 1)
        denoise = core.std.MaskedMerge(denoise, hard_den, lineart)
        out = denoise


        dehalo = MaskedDHA(out, 1.4, 1.4, 0.00, 1.0)
        out = dehalo


        rescale = nnedi3_upscale(out, correct_shift=False).resize.Bicubic(1920, 1080, vs.YUV420P16, src_left=0.5, src_top=0.5)
        out = rescale


        chromafix = xvs.WarpFixChromaBlend(out, 128, depth=4)
        out = chromafix



        deband_mask = detail_mask(get_y(out).rgvs.RemoveGrain(3), 2400, 3000, edgedetect=FDOG())
        deband = vdf.deband.dumb3kdb(out, threshold=45, grain=[48, 24])
        deband = core.std.MaskedMerge(deband, out, deband_mask)
        out = deband

        # return ivtc.resize.Bicubic(1920, 1080).text.FrameProps()
        # return ivtc.resize.Bicubic(1920, 1080), out
        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

    @staticmethod
    def bm3d(clip: vs.VideoNode, sigma: List[float], radius: int, **kwargs: Any) -> vs.VideoNode:
        return cast(vs.VideoNode, eoe.denoise.BM3D(clip, sigma, radius=radius, CUDA=True, **kwargs))


    @staticmethod
    def to_444(c: vs.VideoNode) -> vs.VideoNode:
        return cast(vs.VideoNode, to_444(c, znedi=False))


if __name__ == '__main__':
    filtered = Filtering().main()
    Patch(X265Encoder('x265_settings'), filtered, TV_NHK, [18438, 20556, 20756, 21233, 21657, 21824, 22180, 33926], debug=True).run()
    # X265Encoder('x265_settings').run_enc(filtered, TV_NHK)
    # EztrimCutter(TV_NHK, track=1).run()
    # Mux(TV_NHK).run()
else:
    TV_NHK.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)
