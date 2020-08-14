"""Priconne script"""
__author__ = 'Vardë'

import sys
import os
import subprocess
from functools import partial
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

from cooldegrain import CoolDegrain
import debandshit as dbs
import vardefunc as vdf
import muvsfunc as muvf
import kagefunc as kgf
import G41Fun as gf

from vsutil import depth, get_w, get_y
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

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
    output: str
    chapter: str
    output_final: str


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    src_clip = core.lsmas.LWLibavSource(src, prefer_hw=0, ff_loglevel=3)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.264'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][200807][CYGX-00001]Princess Connect! Re_Dive Vol.1\BD_VIDEO\BDMV\STREAM\00003', 0, -26)
JPBD_NCOP = infos_bd(r'[BDMV][200807][CYGX-00001]Princess Connect! Re_Dive Vol.1\BD_VIDEO\BDMV\STREAM\00007', 0, -24)
JPBD_NCED = infos_bd(r'[BDMV][200807][CYGX-00001]Princess Connect! Re_Dive Vol.1\BD_VIDEO\BDMV\STREAM\00008', 0, -24)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    src += src[-1]



    h = 882
    w = get_w(h)
    kernel = 'lanczos'
    taps = 5
    opstart, opend = 744, 2900
    edstart, edend = 31768, 33925



    denoise = CoolDegrain(src, tr=1, thsad=24, blksize=8, overlap=4, plane=4)
    out = denoise



    luma = get_y(out)

    descale = kgf.get_descale_filter(kernel, taps=taps)(depth(luma, 32), w, h)
    upscale = vdf.fsrcnnx_upscale(depth(descale, 16), src.width, src.height, 'Shaders/FSRCNNX_x2_56-16-4-1.glsl',
                                  partial(core.resize.Bicubic, filter_param_a=0, filter_param_b=0))
    merge = vdf.merge_chroma(upscale, out)
    out = merge




    unwarp = muvf.SharpAAMcmod(out, dark=0.4, thin=-2.5, sharp=20)
    out = unwarp

    dehalo = gf.MaskedDHA(out, rx=1.75, ry=1.75, darkstr=0.05, brightstr=0.95, maskpull=60, maskpush=160)
    out = dehalo



    antialias = lvf.sraa(out, 1.4, rep=7, downscaler=core.resize.Bicubic)
    out = antialias



    deband_mask = lvf.denoise.detail_mask(out, brz_a=2000, brz_b=1000)

    deband = dbs.f3kpf(out, 17, 30, 30)

    deband = core.std.MaskedMerge(deband, out, deband_mask)
    deband = core.neo_f3kdb.Deband(deband, preset='depth', grainy=24, grainc=18, keep_tv_range=True)
    out = deband




    grain = kgf.adaptive_grain(out, 0.2, luma_scaling=14)
    out = grain




    rescale_mask = vdf.drm(src, h, kernel, taps, mthr=40, sw=4, sh=4)
    creditless_mask = core.std.Expr([
        vdf.dcm(src, src[opstart:opend+1], JPBD_NCOP.src_cut[:opend-opstart+1],
                opstart, opend, 2, 2).std.Deflate(),
        vdf.dcm(src, src[edstart:edend+1], JPBD_NCED.src_cut[:edend-edstart+1],
                edstart, edend, 2, 2).std.Deflate()], 'x y +')

    credit = out
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, src, rescale_mask, 0),
                     [(opstart+425, opstart+698), (33972, src.num_frames-1)])
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, src, creditless_mask, 0),
                     [(opstart, opend), (edstart, edend)])
    credit = lvf.rfs(credit, src, [(18917, 18964)])
    out = credit


    return depth(out, 10)


X264_ARGS = dict(
    threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='tesa', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=240, min_keyint=23, rc_lookahead=72, crf=14, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    vdf.encode(clip, 'x264', JPBD.output, **X264_ARGS)

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src, '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, JPBD.a_src_cut.format(1))

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
    encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    f = open("tags_aac.xml", 'w')
    f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                  '</Tag>', '</Tags>'])
    f.close()

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--track-name', '0:AVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # Clean up
    files = [JPBD.a_src, JPBD.a_src_cut.format(1),
             JPBD.a_enc_cut.format(1), 'tags_aac.xml']
    for file in files:
        if os.path.exists(file):
            os.remove(file)

if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
