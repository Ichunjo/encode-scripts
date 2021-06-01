# noqa
from typing import List, NamedTuple, Tuple

import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vdf
from vsutil import iterate, split

import vapoursynth as vs
core = vs.core


class Mask:
    @staticmethod
    def hardsub_mask(base: vs.VideoNode, sub: str, fontdir: str) -> vs.VideoNode:
        _, masksub = core.sub.TextFile(base, sub, fontdir=fontdir, blend=False)
        masksub = masksub.std.Binarize(1)
        masksub = hvf.mt_expand_multi(masksub, 'ellipse', sw=6, sh=4)
        masksub = hvf.mt_inflate_multi(masksub, radius=4).std.Convolution([1] * 9)
        return masksub

    @staticmethod
    def ringing_mask(clip: vs.VideoNode) -> vs.VideoNode:
        linemask = vdf.mask.FreyChenG41().get_mask(clip, lthr=5000, multi=1.5)
        linemask = iterate(linemask, lambda c: core.rgvs.RemoveGrain(c, 2), 3)
        linemask = iterate(linemask, lambda c: core.rgvs.RemoveGrain(c, 3), 2)
        linemask = linemask.std.Maximum().std.Minimum()
        linemask = core.std.Merge(linemask, linemask.std.Minimum(), 0.25)

        ringing_mask = iterate(linemask, core.std.Maximum, 3).std.Inflate()
        ringing_mask = core.std.Expr([ringing_mask, linemask], 'x y -')
        return ringing_mask

    @staticmethod
    def diff_mask(clips: Tuple[vs.VideoNode, vs.VideoNode], brz: float) -> vs.VideoNode:
        mask = core.std.Expr(clips, f'x y 256 * - abs {brz} > 65535 0 ?', vs.YUV420P16)
        mask = iterate(mask, core.std.Maximum, 4).std.Convolution([1]*9)
        return mask

    @staticmethod
    def credits_mask(ep: vs.VideoNode, ncop: vs.VideoNode, nced: vs.VideoNode,
                     opstart: int, opend: int, edstart: int, edend: int) -> vs.VideoNode:
        args = dict(thr=25 << 8, expand=6, prefilter=False)
        op_mask = vdf.mask.Difference().creditless(
            ep, ep[opstart:opend+1], ncop[:opend-opstart+1], opstart, **args)
        ed_mask = vdf.mask.Difference().creditless(
            ep, ep[edstart:edend+1], nced[:edend-edstart+1], edstart, **args)

        mask = core.std.Expr([op_mask, ed_mask], 'x y +').std.Deflate()

        return mask

    @staticmethod
    def deband_mask(clip: vs.VideoNode, kirsch_brz: Tuple[int, int, int], rng_brz: Tuple[int, int, int]) -> vs.VideoNode:
        prefilter = core.bilateral.Bilateral(clip, sigmaS=2.5, sigmaR=0.025)

        kirsch = vdf.mask.Kirsch().get_mask(prefilter).std.Binarize(kirsch_brz)
        rng = lvf.mask.range_mask(prefilter, 4, 4).std.Binarize(rng_brz)
        kirsch, rng = [c.resize.Bilinear(format=vs.YUV444P16) for c in [kirsch, rng]]

        mask = core.std.Expr(split(kirsch) + split(rng), vdf.util.max_expr(6))

        return mask.rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)


class Credit(NamedTuple):
    range_frames: List[Tuple[int, int]]
    mask: vs.VideoNode
