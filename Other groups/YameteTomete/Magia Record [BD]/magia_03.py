"""Magia Record script"""
__author__ = 'Vardë'

import sys
from typing import NamedTuple
from functools import partial
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from vsutil import depth, vs, core, get_y, iterate

from cooldegrain import CoolDegrain
from G41Fun import Tweak, mClean
import debandshit as dbs
import vardefunc as vdf
import kagefunc as kgf
import lvsfunc as lvf

X264 = r'C:\Encode Stuff\x264_tmod_r3007\mcf\x64\x264_x64.exe'

class InfosBD(NamedTuple):
    path: str
    src: str
    src_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    src_cut: vs.VideoNode
    a_src: str
    a_src_cut: str
    a_enc_cut: str
    name: str
    qpfile: str
    output: str
    chapter: str
    output_final: str


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    src_clip = lvf.src(path + '.m2ts')
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.mka'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = sys.argv[0][:-3]
    qpfile = name + '_qpfile.log'
    output = name + '.264'
    chapter = 'chapters/magia_' + name[-2:] + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end, src_cut, a_src, a_src_cut, a_enc_cut,
                   name, qpfile, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][200304][Magia Record][Vol.1]\BD_VIDEO\BDMV\STREAM\00003', 0, -48)
BLANK = 'blank.wav'
X264_ARGS = dict(
    qpfile=JPBD.qpfile, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='tesa', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=360, min_keyint=12, rc_lookahead=72, crf=14.75, qcomp=0.7, aq_mode=3, aq_strength=0.95
)

def do_filter():
    """Vapoursynth filtering"""
    def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1)-> vs.VideoNode:
        bits = clip.format.bits_per_sample - 8
        thr = strength * (1 >> bits)

        luma = get_y(clip)

        def _strong(clip: vs.VideoNode)-> vs.VideoNode:
            args = dict(alpha=0.25, beta=0.5, gamma=40, nrad=2, mdis=20, vcheck=3)
            clip = core.eedi3m.EEDI3(clip, 1, True, **args).std.Transpose()
            clip = core.eedi3m.EEDI3(clip, 1, True, **args).std.Transpose()
            return core.resize.Spline36(clip, luma.width, luma.height, src_left=-.5, src_top=-.5)

        def _weak(clip: vs.VideoNode)-> vs.VideoNode:
            args = dict(nsize=3, nns=2, qual=2)
            clip = core.znedi3.nnedi3(clip, 1, True, **args).std.Transpose()
            clip = core.znedi3.nnedi3(clip, 1, True, **args).std.Transpose()
            return core.resize.Spline36(clip, luma.width, luma.height, src_left=-.5, src_top=-.5)

        clip_aa = core.std.Expr([_strong(luma), _weak(luma), luma],
                                'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr))
        return vdf.merge_chroma(clip_aa, clip)


    def _perform_endcard(path: str, ref: vs.VideoNode)-> vs.VideoNode:
        endcard = lvf.src(path).std.AssumeFPS(ref)
        endcard = core.std.CropRel(endcard, left=14, top=12, right=18, bottom=23)
        endcard = core.resize.Bicubic(endcard, ref.width, ref.height, vs.RGBS, dither_type='error_diffusion')

        endcard = iterate(endcard, partial(core.w2xc.Waifu2x, noise=3, scale=1, photo=True), 2)

        endcard = core.resize.Bicubic(endcard, format=vs.YUV444PS, matrix_s='709', dither_type='error_diffusion')

        return Tweak(endcard, sat=1.2, bright=-0.05, cont=1.2)

    src = JPBD.src_cut
    src = depth(src, 16)
    edstart = 32012

    denoise_a = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=4)
    denoise_b = CoolDegrain(src, tr=3, thsad=96, blksize=8, overlap=4, plane=4)
    denoise = lvf.rfs(denoise_a, denoise_b, [(edstart+1870, edstart+1900)])

    antialias_a = _nneedi3_clamp(denoise)
    antialias_b = lvf.sraa(denoise, 2, rep=13)
    antialias = lvf.rfs(antialias_a, antialias_b, [(14667, 14696)])


    predenoise = mClean(antialias, thSAD=200, chroma=False)
    detail_mask = lvf.denoise.detail_mask(predenoise, rad=2, radc=2, brz_a=3250, brz_b=1250)
    adapt_mask = core.adg.Mask(predenoise.std.PlaneStats(), 40)
    ret_mask = kgf.retinex_edgemask(predenoise).std.Binarize(9250).std.Median().std.Inflate()
    line_mask = core.std.Expr([detail_mask, ret_mask], 'x y max')


    deband_a = core.neo_f3kdb.Deband(antialias, 17, 42, 42, 42, 12, 0, sample_mode=4, keep_tv_range=True)
    deband_a = core.std.MaskedMerge(deband_a, antialias, line_mask)
    deband_b = core.neo_f3kdb.Deband(deband_a, 18, 48, 48, 48, 0, 0, sample_mode=2, keep_tv_range=True)
    deband_ca = dbs.f3kbilateral(deband_b, 8, 270, 270)
    deband_cb = dbs.f3kbilateral(deband_b, 12, 110, 110)
    deband_c = core.std.MaskedMerge(deband_ca, deband_cb, adapt_mask)

    deband = lvf.rfs(deband_a, deband_b, [(edstart+1870, edstart+1900),
                                          (13041, 13180), (13932, 13949),
                                          (14121, 14168), (14647, 14666)])
    deband = lvf.rfs(deband, deband_c, [(14667, 14669), (14673, 14674), (14676, 14677)])


    grain = kgf.adaptive_grain(deband, 0.25, luma_scaling=10)


    crop = core.std.Crop(grain, 0, 0, 132, 132)
    edgefixe = core.edgefixer.ContinuityFixer(crop, 0, [2, 1, 1], 0, [1, 1, 1])
    borders = core.std.AddBorders(edgefixe, 0, 0, 132, 132)
    borders = lvf.rfs(grain, borders, [(0, 2905)])



    endcard = _perform_endcard('[BDMV][200304][Magia Record][Vol.1]/Scans/endcard3_front_descreen.png', src)
    endcard_length = 117
    final = core.std.Splice([borders, endcard * endcard_length], True)
    final = core.resize.Bicubic(final, format=vs.YUV420P10, dither_type='error_diffusion')
    final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

    return depth(final, 10), endcard_length


def do_encode(data):
    """Compression with x264"""
    filtered = data[0]
    endcard_length = data[1]

    print('Qpfile generating')
    vdf.gk(JPBD.src_cut, JPBD.qpfile)

    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, JPBD.output, **X264_ARGS)

    print('\n\n\nAudio extraction')
    mka = MKVFile()
    mka.add_track(MKVTrack(JPBD.src, 1))
    mka.mux(JPBD.a_src)

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, mkvextract_path='mkvextract')
    eztrim(JPBD.src_clip, (0, endcard_length), 'blank.wav', mkvextract_path='mkvextract')

    print('\n\n\nAudio encoding')
    for i in range(1, len(mka.tracks) + 1):
        qaac_args = ['qaac', JPBD.a_src_cut.format(i), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(i)]
        vdf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')
    qaac_args = ['qaac', 'blank_cut_track_1.wav', '-V', '127', '--no-delay', '-o', 'blank_cut.m4a']
    vdf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--language', '0:jpn', JPBD.output,
                '--language', '0:jpn', JPBD.a_enc_cut.format(1), '+', 'blank_cut.m4a',
                '--chapter-language', 'jpn', '--chapters', JPBD.chapter
                ]
    vdf.subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    DATA = do_filter()
    do_encode(DATA)
