"""Gotoubun FF script"""
__author__ = 'Vardë'

from pathlib import Path

import G41Fun as gf
import havsfunc as hvf
import vapoursynth as vs
import vardefunc as vdf
from adptvgrnMod import sizedgrn
from vardautomation import (AudioCutter, AudioEncoder, BasicTool, FileInfo,
                            PresetAAC, PresetBD, X265Encoder)
from vsutil import depth, get_y

from gotoubun_common import AA, Denoise, EncodeGoBrrr, Mask

core = vs.core


NUM = __file__[-5:-3]

JPBD = FileInfo(r'[BDMV][210421][Gotoubun no Hanayome ∬][Vol.2]\BDMV\STREAM\00003.m2ts', 0, None,
                preset=[PresetBD, PresetAAC])

ENCODER = X265Encoder('x265', Path('gotoubun_common/x265_settings'),
                      progress_update=lambda v, e: print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end=""))

A_EXTRACTER = BasicTool('eac3to', [JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL'])
A_CUTTER = AudioCutter(JPBD, track=1)
A_ENCODER = AudioEncoder('qaac', Path('gotoubun_common/qaac_settings'), JPBD, track=1)


class Filtering():
    def main(self) -> vs.VideoNode:
        """Vapoursynth filtering"""
        src = JPBD.clip_cut
        src = depth(src, 16)


        ring_mask = Mask.ringing_mask(get_y(src))

        dering_src = hvf.EdgeCleaner(src, 20, hot=True)
        out = dering_src


        ref = Denoise.ref_denoise(get_y(out), tr=1)
        denoise = Denoise.hybrid_denoise(
            depth(out, 32), knlm_h=0.15, sigma=1,
            knlm_args=dict(d=1, a=3, s=3), bm3d_args=dict(ref=depth(ref, 32))
        )
        denoise = depth(denoise, 16)
        out = denoise



        pre = get_y(out)


        # Remove haloing
        dehalo = gf.MaskedDHA(pre, rx=1.4, ry=1.4, darkstr=0.1, brightstr=1.0, maskpull=40, maskpush=255)
        bila = core.bilateral.Bilateral(dehalo, sigmaS=1, sigmaR=0.08)
        dehalorp = core.rgvs.Repair(dehalo, bila, 13)
        dehalo = core.std.MaskedMerge(dehalo, dehalorp, ring_mask.std.Maximum(), 0)
        dehalo = core.std.Expr([dehalo, pre], 'x y min')
        out = dehalo




        # Antialiasing sraaaaaaaaaa
        lineart = vdf.mask.ExPrewitt().get_mask(out, 4000, 7500).std.Convolution([1] * 25)
        aaa = AA().upscaled_sraaaaaaaaaa(out, height=1620)
        out = aaa




        unwarp = self.line_darkening(out, 0.075).warp.AWarpSharp2(depth=-1)
        out = unwarp


        motion = hvf.SMDegrain(out, tr=3, thSADC=600, RefineMotion=True)
        masked = core.std.MaskedMerge(out, motion, lineart)
        out = masked




        mergechroma = vdf.misc.merge_chroma(out, denoise)
        out = mergechroma


        db_mask = Mask.deband_mask(out, (4000, 4000, 4000), (2000, 2000, 2000))

        deband_1 = vdf.deband.dumb3kdb(out, 17, [36, 48])
        deband_2 = vdf.deband.dumb3kdb(out, 17, 30, sample_mode=4, use_neo=True)

        th_lo, th_hi = 22 << 8, 28 << 8
        strength = f'{th_hi} x - {th_hi} {th_lo} - /'
        deband = core.std.Expr(
            [out.std.Convolution([1]*9), deband_1, deband_2],
            [f'x {th_lo} > x {th_hi} < and z ' + strength + ' * y 1 ' + strength + f' - * + x {th_lo} <= z y ? ?',
             'x y * sqrt x z * sqrt * y * z * 0.25 pow'])

        deband = core.std.MaskedMerge(deband, out, db_mask)
        out = deband




        ref = get_y(out).std.PlaneStats()
        adgmask_a = ref.adg.Mask(25)
        adgmask_b = ref.adg.Mask(10)

        stgrain_a = core.grain.Add(out, 0.1, 0, seed=333, constant=True)
        stgrain_a = core.std.MaskedMerge(out, stgrain_a, adgmask_b.std.Invert())

        stgrain_b = sizedgrn(out, 0.2, 0.1, 1.15, sharp=80, static=True, fade_edges=False, protect_neutral=False, seed=333)
        stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_b)
        stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_a.std.Invert())
        stgrain = core.std.MergeDiff(stgrain_b, out.std.MakeDiff(stgrain_a))

        dygrain = sizedgrn(out, 0.3, 0.1, 1.25, sharp=80, static=False, fade_edges=False, protect_neutral=False, seed=333)
        dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
        grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
        out = grain


        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])



    @staticmethod
    def line_darkening(clip: vs.VideoNode, strength: float = 0.2, **kwargs) -> vs.VideoNode:
        """Darken lineart through Toon

        Args:
            clip (vs.VideoNode): Source clip.
            strength (float, optional): Strength in Toon. Defaults to 0.2.

        Returns:
            vs.VideoNode: Darked clip.
        """
        darken = hvf.Toon(clip, strength, **kwargs)
        darken_mask = core.std.Expr(
            [core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
             core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
            ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (clip.format.bits_per_sample-1),
                                                            peak=(1 << clip.format.bits_per_sample)-1)])
        return core.std.MaskedMerge(clip, darken, darken_mask)


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = EncodeGoBrrr(filtered, JPBD, ENCODER, None, A_EXTRACTER, A_CUTTER, A_ENCODER)
    brrrr.run()
    brrrr.cleanup()
else:
    JPBD.clip_cut.text.Text('BD').set_output(0)
    # lvf.comparison.stack_planes(JPBD.clip_cut).set_output(1)
    # WEB_AOD.clip_cut.text.Text('AoD').set_output(1)
    # WEB_CRU.clip_cut.text.Text('CR').set_output(2)
    FILTERED = Filtering().main()
    FILTERED.text.Text('Filtering').set_output(3)
    # FILTERED.set_output(3)
    # FILTERED[0].set_output(1)
    # FILTERED[1].set_output(2)
    # FILTERED[2].set_output(3)
