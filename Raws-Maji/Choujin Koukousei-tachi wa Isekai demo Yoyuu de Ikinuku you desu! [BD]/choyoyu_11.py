"""Choyoyu script"""
__author__ = 'Vardë'

import sys
from typing import Callable
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from vsutil import core, vs, depth, get_y, join
from cooldegrain import CoolDegrain

import vardefunc as vdf
import debandshit as dbs
import lvsfunc as lvf

PATH = r'[BDMV] CHOYOYU\CHOYOYU_4\BDMV\STREAM\00003'
SRC = PATH + '.m2ts'
SRC_CLIP = lvf.src(PATH + '.dgi')

FRAME_START, FRAME_END = 0, -24
SRC_CUT = SRC_CLIP[FRAME_START:FRAME_END]

A_SRC = PATH + '.mka'
A_SRC_CUT = PATH + '_cut_track_1.wav'
A_ENC_CUT = PATH + '_track_1.m4a'
NAME = sys.argv[0][:-3]
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
CHAPTER = 'chapters/chapter' + NAME[-2:] + '.txt'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=QPFILE, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:0', me='umh', subme=10, psy_rd='1.0:0.00', merange=24,
    keyint=240, min_keyint=1, rc_lookahead=60, crf=14.85, qcomp=0.7, aq_mode=3, aq_strength=1.0
)

def do_filter():
    """Vapoursynth filtering"""

    def _fsrcnnx(clip: vs.VideoNode) -> vs.VideoNode:
        blank = core.std.BlankClip(clip, format=vs.GRAY16, color=128 << 8)
        clip = join([clip, blank, blank])
        # The chroma is upscaled with box AKA nearest but we don't care since we only need the luma.
        # It's especially faster and speed is the key :^)
        clip = core.placebo.Shader(clip, 'Shaders/FSRCNNX_x2_56-16-4-1.glsl', clip.width*2, clip.height*2, filter='box')
        return get_y(clip)

    def _nnedi(clip: vs.VideoNode) -> vs.VideoNode:
        args = dict(nsize=4, nns=4, qual=2, pscrn=2)
        clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
            .std.Transpose().nnedi3.nnedi3(0, True, **args)
        return core.resize.Spline36(clip, src_top=.5, src_left=.5)

    def _rescale(clip: vs.VideoNode, thr: int, downscaler: Callable[[vs.VideoNode], vs.VideoNode]) -> vs.VideoNode:
        return downscaler(core.std.Merge(_nnedi(clip), _fsrcnnx(clip), thr))

    src = SRC_CUT

    fixedges = lvf.ef(src, [1, 1, 1])

    denoise = CoolDegrain(depth(fixedges, 16), tr=1, thsad=48, thsadc=84, blksize=8, overlap=4, plane=4)


    w, h = 1280, 720
    b, c = vdf.get_bicubic_params('robidoux')
    luma = get_y(denoise)
    descale = depth(core.descale.Debicubic(depth(luma, 32), w, h, b, c), 16)
    rescale = _rescale(descale, 0.55, lambda c: core.resize.Bicubic(c, src.width, src.height, filter_param_a=0, filter_param_b=0))

    line_mask = detail_mask_func(rescale, brz_a=7000, brz_b=2000)
    rescale = core.std.MaskedMerge(luma, rescale, line_mask)



    credit_mask = vdf.drm(luma, b=b, c=c, mode='ellipse', sw=3, sh=3)
    credit = core.std.MaskedMerge(rescale, luma, credit_mask)


    merged = vdf.merge_chroma(credit, denoise)


    deband_mask = detail_mask_func(merged, brz_a=3000, brz_b=1500)
    deband = dbs.f3kpf(merged, 17, 36, 36)
    deband = core.std.MaskedMerge(deband, merged, deband_mask)
    grain = core.neo_f3kdb.Deband(deband, preset='depth', grainy=32, output_depth=10,
                                  dither_algo=3, keep_tv_range=True)

    return grain


def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vdf.gk(SRC_CUT, QPFILE)

    print('\n\n\nVideo encode')
    vdf.encode(filtered, X264, OUTPUT, **X264_ARGS)

    print('\n\n\nAudio extraction')
    mka = MKVFile()
    mka.add_track(MKVTrack(SRC, 1))
    mka.mux(A_SRC)

    print('\n\n\nAudio cut')
    eztrim(SRC_CLIP, (FRAME_START, FRAME_END), A_SRC, mkvextract_path='mkvextract')

    print('\n\n\nAudio encode')
    qaac_args = ['qaac64', A_SRC_CUT, '-V', '127', '--no-delay', '-o', A_ENC_CUT]
    vdf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')


    print('\nFinal mux')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(OUTPUT, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(A_ENC_CUT, language='jpn', default_track=True))
    mkv.chapters(CHAPTER, 'jpn')
    mkv.mux(OUTPUT_FINAL)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
