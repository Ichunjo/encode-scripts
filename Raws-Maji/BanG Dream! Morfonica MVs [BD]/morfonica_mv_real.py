"""Morfonica script"""
__author__ = 'Vardë'

import sys
from pymkv import MKVFile, MKVTrack

from vsutil import get_y, depth, core
from acsuite import eztrim

import mvsfunc as mvf
import vardefunc as vrf
import debandshit as dbs
import lvsfunc as lvf

PATH = r'[200527]Morfonica 1st Single「Daylight -デイライト- 」\BD\BDMV\STREAM\00001'
SRC = PATH + '.m2ts'
SRC_CLIP = lvf.src(SRC)

FRAME_START, FRAME_END = 0, -24
SRC_CUT = SRC_CLIP[FRAME_START:FRAME_END]

A_SRC = PATH + '.wav'
A_SRC_CUT = PATH + '_cut_track_1.wav'
NAME = sys.argv[0][:-3]
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=QPFILE, threads=27, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-1', me='umh', subme=10, psy_rd='0.70:0.00', merange=24,
    keyint=240, min_keyint=1, rc_lookahead=60, crf=16.5, qcomp=0.65, aq_mode=3, aq_strength=0.8
)

def do_filter():
    """Vapoursynth filtering"""
    src = SRC_CUT
    src = depth(src, 16)
    denoise = core.knlm.KNLMeansCL(src, a=2, h=0.8, d=3, device_type='gpu', channels='UV')
    denoise_mask = core.adg.Mask(src.std.PlaneStats(), 2).std.Invert()
    denoise = core.std.MaskedMerge(denoise, mvf.BM3D(denoise, 2), denoise_mask)


    antialias = lvf.sraa(denoise, 1.75, 6, sharp_downscale=True)
    antialias_mask = core.std.Prewitt(get_y(denoise)).std.Maximum()
    antialias = core.std.MaskedMerge(denoise, antialias, antialias_mask)


    preden = core.knlm.KNLMeansCL(antialias, a=2, h=1.5, d=0, device_type='gpu', channels='Y')
    deband_mask = lvf.denoise.detail_mask(preden, brz_a=2500, brz_b=1000)
    deband = dbs.f3kpf(preden, 17, 36, 36)
    diff = core.std.MakeDiff(antialias, preden)
    deband = core.std.MergeDiff(deband, diff)
    deband = core.std.MaskedMerge(deband, antialias, deband_mask)

    final = depth(deband, 10)

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
