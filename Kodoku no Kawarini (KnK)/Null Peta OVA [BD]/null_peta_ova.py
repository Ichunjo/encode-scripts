"""Null Peta OVA script"""
__author__ = 'Vardë'

import sys
from pymkv import MKVFile, MKVTrack

from vsutil import get_y, get_w, depth, core, vs, join, split
from acsuite import eztrim
from cooldegrain import CoolDegrain
from nnedi3_rpow2 import nnedi3_rpow2

import vardefunc as vrf
import debandshit as dbs
import kagefunc as kgf
import lvsfunc as lvf

PATH = r'00017'
SRC = PATH + '.m2ts'
SRC_CLIP = lvf.src(SRC)

FRAME_START, FRAME_END = 0, -48
SRC_CUT = SRC_CLIP[FRAME_START:FRAME_END]

OPSTART, OPEND = 0, 1485

A_SRC = PATH + '.wav'
A_SRC_CUT = PATH + '_cut_track_1.wav'
A_ENC_CUT = PATH + '.m4a'
NAME = sys.argv[0][:-3]
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
CHAPTER = 'chapter_' + NAME[-3:] + '.txt'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=QPFILE, threads=27, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-2', me='umh', subme=10, psy_rd='1.0:0.00', merange=24,
    keyint=240, min_keyint=1, rc_lookahead=60, crf=14.5, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_filter():
    """Vapoursynth filtering"""

    def _fsrcnnx(clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
        blank = core.std.BlankClip(clip, format=vs.GRAY16, color=128 << 8)
        clip = join([clip, blank, blank])
        clip = core.placebo.Shader(clip, 'FSRCNNX_x2_56-16-4-1.glsl',
                                   clip.width * 2, clip.height * 2)
        return core.resize.Spline36(get_y(clip), width, height)

    src = SRC_CUT

    fe = lvf.ef(src, [1, 1, 1])
    fe = depth(fe, 16)



    h = 864
    w = get_w(864)
    b, c = 0, 1/2
    luma = get_y(fe)
    descale = core.descale.Debicubic(depth(luma, 32), w, h, b, c)
    descale = depth(descale, 16)

    rescale_a = nnedi3_rpow2(descale, 2, src.width, src.height, nns=4, qual=2, pscrn=2)
    rescale_b = _fsrcnnx(descale, src.width, src.height)
    rescale = core.std.Merge(rescale_a, rescale_b, 0.75)

    rescale_mask = vrf.drm(fe, 864, b=b, c=c, mthr=80, sw=4, sh=4)
    rescale = core.std.MaskedMerge(rescale, luma, rescale_mask)

    rescale = lvf.rfs(rescale, luma, [(OPSTART+483, OPSTART+721), (OPSTART+822, OPSTART+1083)])
    merge = core.std.ShufflePlanes([rescale, fe], [0, 1, 2], vs.YUV)



    antialias = join([lvf.sraa(plane) for plane in split(merge)])
    antialias = lvf.rfs(merge, antialias, [(2836, 2870)])



    denoise = core.knlm.KNLMeansCL(antialias, a=2, h=0.65, d=0, device_type='gpu', channels='UV')



    preden = CoolDegrain(denoise, tr=2, thsad=60, blksize=8, overlap=4, plane=4)
    diff = core.std.MakeDiff(denoise, preden)
    deband_mask = lvf.denoise.detail_mask(preden, brz_a=3000, brz_b=1500)

    deband_a = dbs.f3kpf(preden, 17, 42, 42)
    deband_b = core.placebo.Deband(preden, radius=17, threshold=6, iterations=1, grain=0, planes=1|2|4)
    deband = lvf.rfs(deband_a, deband_b, [(2081, 2216), (2450, 2550), (3418, 3452), (3926, 3926)])

    deband = core.std.MaskedMerge(deband, preden, deband_mask)
    deband = core.std.MergeDiff(deband, diff)


    grain = kgf.adaptive_grain(deband, 0.25, luma_scaling=8)
    final = depth(grain, 10)

    return final


def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vrf.gk(SRC_CUT, QPFILE)

    print('\n\n\nVideo encode')
    vrf.encode(filtered, X264, OUTPUT, **X264_ARGS)

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', SRC, '2:', A_SRC, '-log=NUL']
    vrf.subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cut')
    eztrim(SRC_CLIP, (FRAME_START, FRAME_END), A_SRC, mkvextract_path='mkvextract')

    print('\n\n\nAudio encode')
    qaac_args = ['qaac64', A_SRC_CUT, '-V', '127', '--no-delay', '-o', A_ENC_CUT]
    vrf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal mux')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(OUTPUT, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(A_ENC_CUT, language='jpn', default_track=True))
    mkv.chapters(CHAPTER, 'jpn')
    mkv.mux(OUTPUT_FINAL)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED[0])
