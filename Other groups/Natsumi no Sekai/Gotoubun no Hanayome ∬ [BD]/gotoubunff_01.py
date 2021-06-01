"""Gotoubun FF script"""
__author__ = 'Vardë'

from pathlib import Path
from typing import List

import G41Fun as gf
import havsfunc as hvf
import lvsfunc as lvf
import vapoursynth as vs
import vardefunc as vdf
from adptvgrnMod import sizedgrn
from vardautomation import (AudioCutter, AudioEncoder, BasicTool, FileInfo,
                            LosslessEncoder, PresetAAC, PresetBD, X265Encoder)
from vsutil import depth, get_y, insert_clip

from gotoubun_common import AA, Denoise, EncodeGoBrrr, Mask

core = vs.core


NUM = __file__[-5:-3]

JPBD = FileInfo(r'[BDMV][210317][Gotoubun no Hanayome ∬][Vol.1]\BDMV\STREAM\00005.m2ts', 0, 34647,
                preset=[PresetBD, PresetAAC])
JPBD_NCOP = FileInfo(r'[BDMV][210317][Gotoubun no Hanayome ∬][Vol.1]\BDMV\STREAM\00009.m2ts', 0, 2158)
JPBD_NCED = FileInfo(r'[BDMV][210421][Gotoubun no Hanayome ∬][Vol.2]\BDMV\STREAM\00009.m2ts', 0, -24)

WEB_AOD = FileInfo(fr'5-toubun no Hanayome S02 (The Quintessential Quintuplets S02) [AoD]\5-toubun no Hanayome S02E{NUM} [1080p+][AAC][JapDub][GerSub][Web-DL].mkv',)
WEB_CRU = FileInfo(fr'5-toubun no Hanayome S02 (The Quintessential Quintuplets S02) [CR]\5-toubun no Hanayome S02E{NUM} [1080p][AAC][JapDub][GerEngSub][Web-DL].mkv')
SUB = fr'5-toubun_no_Hanayome_subs\ger\[FeelsBadSubs] 5-toubun no Hanayome ∬ - {NUM} [1080p].4.ger.ass'
FONTDIR = '5-toubun_no_Hanayome_subs/_fonts'


OPSTART, OPEND = 528, 2685
EDSTART, EDEND = 32370, 34526

ENC_LOSS = LosslessEncoder('nvencc', Path('gotoubun_common/nvenc_settings'), progress_update=None)

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
        src_cru = WEB_CRU.clip_cut
        src_aod = WEB_AOD.clip_cut


        # Dehardsubbing using .ass file
        masksub = Mask.hardsub_mask(src_aod.std.BlankClip(), SUB, FONTDIR)
        dehardsub = core.std.MaskedMerge(src_aod, src_cru, masksub)
        src_web = dehardsub


        # AoD is poorly rescaled
        src_y, src_cru_y, src_web_y = map(get_y, [src, src_cru, src_web])

        thr = 10
        src_web_y = core.std.Expr([src_web_y, src_cru_y], f'x y - abs {thr} < x y * sqrt y ?')
        src_web_y = depth(src_web_y, 16)

        ringrid = core.rgvs.Repair(src_web_y, src_web_y.std.Convolution([1]*9), 11)
        ringrid = core.std.Expr([ringrid, src_web_y], 'x y min')

        # Remove ringing with AoD+CR
        ring_mask = Mask.ringing_mask(src_y)
        bdweb = core.std.MaskedMerge(src_y, ringrid, ring_mask)
        out = vdf.misc.merge_chroma(bdweb, src)



        # Restore BD changes
        dering_src = hvf.EdgeCleaner(src, 20, hot=True)

        diff_mask = Mask.diff_mask((src, src_cru), brz=4750.0)
        bdchanges = core.std.MaskedMerge(out, dering_src, diff_mask)
        bdchanges = lvf.rfs(out, bdchanges, [(17008, 17085), (17200, 17283)])
        bdchanges = lvf.rfs(bdchanges, src, [(17551, 17730)])
        out = bdchanges


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


        ref = depth(src_cru, 16)
        cred_mask = Mask.credits_mask(src, JPBD_NCOP.clip_cut, JPBD_NCED.clip_cut, OPSTART, OPEND, EDSTART, EDEND)
        cred = out
        cred = lvf.rfs(cred, core.std.MaskedMerge(cred, ref, cred_mask, 0), [(OPSTART, OPEND), (EDSTART, EDEND)])
        out = cred



        db_mask = Mask.deband_mask(out, (4000, 4000, 4000), (2000, 2000, 2000))

        deband_1 = vdf.deband.dumb3kdb(out, 17, [36, 48])
        deband_2 = vdf.deband.dumb3kdb(out, 17, 30, sample_mode=4, use_neo=True)
        deband_a = vdf.placebo.deband(out, 22, threshold=8, iterations=3, grain=0)


        th_lo, th_hi = 22 << 8, 28 << 8
        strength = f'{th_hi} x - {th_hi} {th_lo} - /'
        deband = core.std.Expr(
            [out.std.Convolution([1]*9), deband_1, deband_2],
            [f'x {th_lo} > x {th_hi} < and z ' + strength + ' * y 1 ' + strength + f' - * + x {th_lo} <= z y ? ?',
             'x y * sqrt x z * sqrt * y * z * 0.25 pow'])

        deband = lvf.rfs(deband, deband_a, [(OPSTART+1382, OPSTART+1433)])
        deband = core.std.MaskedMerge(deband, out, db_mask)
        out = deband



        newpan = self.refine_motion(out[17284:17329], [1, 5, 9, 14, 18, 22, 27, 31, 35], out)
        motion = insert_clip(out, newpan, 17284)
        out = motion





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

    @staticmethod
    def refine_motion(pan: vs.VideoNode, del_frames: List[int], ref: vs.VideoNode) -> vs.VideoNode:
        delfra = core.std.DeleteFrames(pan, del_frames)

        time = pan.num_frames / (ref.fps.numerator / ref.fps.denominator)

        delfra = core.std.AssumeFPS(delfra, fpsnum=delfra.num_frames * 10000, fpsden=round(time * 10000))
        delfra += delfra[-1]

        newpan = gf.JohnFPS(delfra, ref.fps.numerator, ref.fps.denominator, pel=4)
        return newpan


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = EncodeGoBrrr(filtered, JPBD, ENCODER, ENC_LOSS, A_EXTRACTER, A_CUTTER, A_ENCODER)
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
