"""Magia Record script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from typing import NamedTuple
from pathlib import Path

from acsuite import eztrim

from cooldegrain import CoolDegrain
from G41Fun import mClean, Tweak
import vardefunc as vdf
import kagefunc as kgf

from vsutil import depth, get_y
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

X264 = 'x264_tmod'

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
    src_clip = lvf.src(src, ff_loglevel=4)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}'
    name = Path(sys.argv[0]).stem
    qpfile = name + '_qpfile.log'
    output = name + '.264'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end, src_cut, a_src, a_src_cut, a_enc_cut,
                   name, qpfile, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][200902][Magia Record][Vol.5]\BD_VIDEO\BDMV\STREAM\00001', 0, -48)
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
        endcard = core.std.CropRel(endcard, left=4, top=12, right=0, bottom=21)
        endcard = core.resize.Bicubic(endcard, ref.width, ref.height, vs.RGBS, dither_type='error_diffusion')

        endcard = core.w2xc.Waifu2x(endcard, noise=3, scale=1, photo=True)

        endcard = core.resize.Bicubic(endcard, format=vs.YUV444PS, matrix_s='709', dither_type='error_diffusion')
        endcard = lvf.util.quick_resample(endcard, lambda c: core.neo_f3kdb.Deband(c, 15, 36, 36, 36, 24, 24, 4))

        return Tweak(endcard, sat=1.2, bright=-0.05, cont=1.2)

    src = JPBD.src_cut
    src = depth(src, 16)
    edstart = 31769

    denoise_a = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=4)
    denoise_b = CoolDegrain(src, tr=3, thsad=96, blksize=8, overlap=4, plane=4)
    denoise = lvf.rfs(denoise_a, denoise_b, [(edstart+1870, edstart+1900)])

    antialias = _nneedi3_clamp(denoise)


    predenoise = mClean(antialias, thSAD=200, chroma=False)
    detail_mask = detail_mask_func(predenoise, rad=2, radc=2, brz_a=3250, brz_b=1250)
    ret_mask = kgf.retinex_edgemask(predenoise).std.Binarize(9250).std.Median().std.Inflate()
    line_mask = core.std.Expr([detail_mask, ret_mask], 'x y max')


    deband_a = core.neo_f3kdb.Deband(antialias, 17, 42, 42, 42, 12, 0, sample_mode=4, keep_tv_range=True)
    deband_a = core.std.MaskedMerge(deband_a, antialias, line_mask)
    deband_b = core.neo_f3kdb.Deband(deband_a, 18, 48, 48, 48, 0, 0, sample_mode=2, keep_tv_range=True)

    deband = lvf.rfs(deband_a, deband_b, [(edstart+1870, edstart+1900)])

    grain = kgf.adaptive_grain(deband, 0.25, luma_scaling=10)


    # endcard = _perform_endcard(r'[BDMV][200902][Magia Record][Vol.5]\Scans\endcard11_front_good.png', src)
    endcard = lvf.src(r'[BDMV][200902][Magia Record][Vol.5]\Scans\endcard11_front_good.png').std.AssumeFPS(src)
    endcard = core.resize.Bicubic(endcard, format=vs.YUV444PS, matrix_s='709', dither_type='error_diffusion')
    endcard_length = 119
    final = core.std.Splice([grain, endcard * endcard_length], True)
    final = core.resize.Bicubic(final, format=vs.YUV420P10, dither_type='error_diffusion')
    final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

    return depth(final, 10), endcard_length


def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vdf.gk(JPBD.src_cut, JPBD.qpfile)



    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, JPBD.output, **X264_ARGS)



    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src,
                   '2:', JPBD.a_src.format(1),
                   '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')



    print('\n\n\nAudio cutting & encoding')
    for i in range(1, 1+1):
        eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end),
               JPBD.a_src.format(i),
               JPBD.a_src_cut.format(i))

        # Encode in AAC
        qaac_args = ['qaac', JPBD.a_src_cut.format(i),
                     '-V', '127', '--no-delay',
                     '-o', JPBD.a_enc_cut.format(i) + '.m4a']
        subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

        # Encode in Opus
        opus_args = ['opusenc', '--bitrate', '192',
                     JPBD.a_src_cut.format(i),
                     JPBD.a_enc_cut.format(i) + '.opus']
        subprocess.run(opus_args, text=True, check=True, encoding='utf-8')

    # Blanck cut and encode
    eztrim(JPBD.src_clip, (0, ENDCARD_LENGTH),
           'blank.wav', 'blank_cut.wav')
    qaac_args = ['qaac', 'blank_cut.wav',
                 '-V', '127', '--no-delay',
                 '-o', 'blank_cut.m4a']
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    opus_args = ['opusenc', '--bitrate', '192',
                 'blank_cut.wav',
                 'blank_cut.opus']
    subprocess.run(opus_args, text=True, check=True, encoding='utf-8')

    # Recup encoder name
    ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder',
                    '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1) + '.m4a']
    encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    fil = open("tags_aac.xml", 'w')
    fil.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                  '</Tag>', '</Tags>'])
    fil.close()

    opus_args = ['opusenc', '-V']
    encoder_name = subprocess.check_output(opus_args, shell=True, encoding='utf-8').splitlines()[0]
    fil = open("tags_opus.xml", 'w')
    fil.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}, VBR 192k</String>', '</Simple>',
                  '</Tag>', '</Tags>'])
    fil.close()



    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.name + '_aac.mkv',
                '--language', '0:jpn', '--track-name', '0:AVC BDRip by Vardë@YameteTomete', JPBD.output,
                '--tags', '0:tags_aac.xml', '--language', '0:jpn', '--track-name', '0:AAC 2.0', JPBD.a_enc_cut.format(1) + '.m4a', '+', 'blank_cut.m4a',
                '--chapter-language', 'en', '--chapters', JPBD.chapter
                ]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    mkv_args = ['mkvmerge', '-o', JPBD.name + '_opus.mkv',
                '--language', '0:jpn', '--track-name', '0:AVC BDRip by Vardë@YameteTomete', JPBD.output,
                '--tags', '0:tags_opus.xml', '--language', '0:jpn', '--track-name', '0:Opus 2.0', JPBD.a_enc_cut.format(1) + '.opus', '+', 'blank_cut.opus',
                '--chapter-language', 'en', '--chapters', JPBD.chapter
                ]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')



    # Clean up
    files = ['tags_aac.xml', 'tags_opus.xml',
             'blank_cut.wav', 'blank_cut.m4a', 'blank_cut.opus']
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    # for i in range(1, 2+1):
    for i in range(1, 1+1):
        files = [JPBD.a_src.format(i), JPBD.a_src_cut.format(i),
                 JPBD.a_enc_cut.format(i) + '.m4a', JPBD.a_enc_cut.format(i) + '.opus']
        for file in files:
            if os.path.exists(file):
                os.remove(file)



if __name__ == '__main__':
    FILTERED, ENDCARD_LENGTH = do_filter()
    do_encode(FILTERED)
