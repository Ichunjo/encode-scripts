"""Dazzling White Town.py"""
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
import havsfunc as hvf
import kagefunc as kgf
import G41Fun as gf
import placebo
import xvs

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
    a_enc_cut = path + '_track_{}.flac'
    name = Path(sys.argv[0]).stem
    output = name + '.264'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)


JPBD = infos_bd(r'[EAC][200819][SINGLE][Saint Snow][Dazzling White Town][LACM-14934][WAV+CUE+LOG+PNG+BDMV]\BDMV\STREAM\00001', 480, -24)

def do_filter():
    src = JPBD.src_cut

    fixedges = lvf.ef(src, [2, 1, 1])
    fixedges = depth(fixedges, 16)
    out = fixedges


    h = 720
    w = get_w(h)
    kernel = 'bilinear'


    denoise = CoolDegrain(out, tr=1, thsad=24, blksize=8, overlap=4, plane=4)
    out = denoise


    luma = get_y(out)

    line_mask = vdf.edge_detect(luma, 'FDOG', 0.05, (1, 1))

    descale = kgf.get_descale_filter(kernel)(depth(luma, 32), w, h)
    rescale = vdf.fsrcnnx_upscale(depth(descale, 16), None, src.height, 'shaders/FSRCNNX_x2_56-16-4-1.glsl',
                                  partial(muvf.SSIM_downsample, kernel='Bicubic'))
    rescale = core.std.MaskedMerge(luma, rescale, line_mask)


    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)


    # Slight sharp though CAS
    sharp = hvf.LSFmod(out, strength=95, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp

    dering = gf.HQDeringmod(out, thr=16, darkthr=0.1)
    out = dering

    warp = xvs.WarpFixChromaBlend(out, thresh=36, depth=6)
    out = warp



    deband_mask = detail_mask_func(out, brz_a=2500, brz_b=1500)
    deband = dbs.f3kpf(out, 17, 36, 36)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband


    grain = placebo.deband(out, iterations=0, grain=6, chroma=False)
    grain_mask = core.adg.Mask(out.std.PlaneStats(), 14).std.Expr(f'x x {128<<8} - 0.25 * +')
    grain = core.std.MaskedMerge(out, grain, grain_mask)
    out = grain

    rescale_mask = vdf.drm(luma, h, kernel, sw=4, sh=4)
    ref = fixedges
    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, rescale_mask, 0), [(12805, src.num_frames-1)])
    out = credit


    return depth(out, 10)



X264_ARGS = dict(
    threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-2', me='tesa', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=240, min_keyint=23, rc_lookahead=96, crf=14, qcomp=0.7, aq_mode=3, aq_strength=1.0
)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    if not os.path.exists(JPBD.output):
        print('\n\n\nVideo encoding')
        vdf.encode(clip, 'x264', JPBD.output, **X264_ARGS, frames=clip.num_frames)

    if not os.path.exists(JPBD.a_src):
        print('\n\n\nAudio extraction')
        eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src, '-log=NUL']
        subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    if not os.path.exists(JPBD.a_src_cut.format(1)):
        print('\n\n\nAudio cutting')
        eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, JPBD.a_src_cut.format(1))

    if not os.path.exists(JPBD.a_enc_cut.format(1)):
        print('\n\n\nAudio encoding')
        qaac_args = ['ffmpeg', '-i', JPBD.a_src_cut.format(1), '-compression_level', '12',
                     '-lpc_type', 'cholesky', '-lpc_passes', '3', '-exact_rice_parameters', '1',
                     JPBD.a_enc_cut.format(1)]
        subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    if not os.path.exists(JPBD.output_final):
        print('\nFinal muxing')
        mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                    '--track-name', '0:AVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                    '--track-name', '0:FLAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1)]
        subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # Clean up
    files = [JPBD.a_src, JPBD.a_src_cut.format(1), JPBD.a_enc_cut.format(1)]
    for file in files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    FILTERED = do_filter()
    do_encode(FILTERED)
