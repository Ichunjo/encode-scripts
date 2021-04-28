"""Morfonica script"""
__author__ = 'Vardë'

import sys
from pymkv import MKVFile, MKVTrack

from vsutil import get_y, get_w, depth, core, vs
from acsuite import eztrim

import vardefunc as vrf
import debandshit as dbs
import modfunc as mdf
import lvsfunc as lvf

PATH = r'[200527]Morfonica 1st Single「Daylight -デイライト- 」\BD\BDMV\STREAM\00000'
SRC = PATH + '.m2ts'
SRC_CLIP = lvf.src(SRC)

FRAME_START, FRAME_END = 0, 3306
SRC_CUT = SRC_CLIP[FRAME_START:FRAME_END]

A_SRC = PATH + '.wav'
A_SRC_CUT = PATH + '_cut_track_1.wav'
NAME = sys.argv[0][:-3]
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=QPFILE, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='umh', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=240, min_keyint=1, rc_lookahead=60, crf=14.75, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_filter():
    """Vapoursynth filtering"""
    def _eedi3_instance(clip, eeargs, nnargs) -> vs.VideoNode:
        return core.eedi3m.EEDI3(clip, 0, True, **eeargs, sclip=_nnedi3_sclip(clip, nnargs))

    def _nnedi3_sclip(clip, nnargs) -> vs.VideoNode:
        return core.nnedi3.nnedi3(clip, 0, True, **nnargs)

    def _rescale(clip, width, height, eeargs, nnargs) -> vs.VideoNode:
        clip = _eedi3_instance(clip, eeargs, nnargs).std.Transpose()
        clip = _eedi3_instance(clip, eeargs, nnargs).std.Transpose()
        return core.resize.Bicubic(clip, width, height, src_left=.5, src_top=.5)

    src = SRC_CUT

    denoise = mdf.hybriddenoise_mod(src, 0.5, 1.5, depth=32)

    h = 806
    w = get_w(h)
    b, c = 0.3782, 0.3109
    eedi3_args = dict(alpha=.2, beta=.3, gamma=1000, mdis=20, vcheck=3)
    nnedi3_args = dict(nns=4, nsize=0, qual=2, pscrn=1)

    luma = get_y(denoise)
    descale = core.descale.Debicubic(luma, w, h, b, c)
    scaled = _rescale(descale, src.width, src.height, eedi3_args, nnedi3_args)

    credit_m = vrf.drm(denoise, h, b=b, c=c, sw=4, sh=4)
    scaled = lvf.rfs(scaled, core.std.MaskedMerge(scaled, luma, credit_m), [(47, 134)])
    merged = core.std.ShufflePlanes([scaled, denoise], [0, 1, 2], vs.YUV)
    merged = depth(merged, 16)


    deband_mask = detail_mask_func(merged, brz_a=2500, brz_b=1000)

    deband_a = dbs.f3kpf(merged, 17, 48, 48)
    deband_b = core.placebo.Deband(merged, radius=19, threshold=5, iterations=1, grain=2, planes=1|2|4)
    deband = lvf.rfs(deband_a, deband_b, [(2684, 2743)])
    deband = core.std.MaskedMerge(deband, merged, deband_mask)
    deband = core.placebo.Deband(deband, iterations=0, grain=5, planes=1)

    grain = mdf.adptvgrnMod_mod(deband, 0.3, size=1.25, sharp=60, static=False, luma_scaling=6)

    final = lvf.rfs(grain, depth(src, 16), [(2904, src.num_frames-1)])
    final = depth(final, 10)

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
    flac_args = ['flac', A_SRC_CUT, '-8']
    vrf.subprocess.run(flac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal mux')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(OUTPUT, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(A_SRC_CUT[:-3] + 'flac', language='jpn', default_track=True))
    mkv.mux(OUTPUT_FINAL)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
