"""Priconne script"""
__author__ = 'Vardë'

import sys
import os
import shlex
import subprocess
from functools import partial
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

from cooldegrain import CoolDegrain
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf
import kagefunc as kgf

from _assets.priconnefunc import line_darkening, stabilization
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
    output = name + '.265'
    chapter = '_assets/chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][201106][CYGX-00004]Princess Connect! Re_Dive Vol.4\BD_VIDEO\BDMV\STREAM\00002', 0, -24)
JPBD_NCOP = infos_bd(r'[BDMV][200807][CYGX-00001]Princess Connect! Re_Dive Vol.1\BD_VIDEO\BDMV\STREAM\00007', 0, -24)
JPBD_NCED = infos_bd(r'[BDMV][200807][CYGX-00001]Princess Connect! Re_Dive Vol.1\BD_VIDEO\BDMV\STREAM\00008', 0, -24)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)

    h = 882
    w = get_w(h)
    kernel = 'lanczos'
    taps = 5
    opstart, opend = 1248, 3407
    edstart, edend = 33930, 36087


    denoise = CoolDegrain(src, tr=1, thsad=24, blksize=8, overlap=4, plane=4)
    out = denoise



    luma = get_y(out)

    descale = kgf.get_descale_filter(kernel, taps=taps)(depth(luma, 32), w, h)
    upscale = vdf.fsrcnnx_upscale(depth(descale, 16), src.width, src.height, '_assets/shaders/FSRCNNX_x2_56-16-4-1.glsl',
                                  partial(core.resize.Bicubic, filter_param_a=0, filter_param_b=0))
    out = upscale



    unwarp = line_darkening(out, 0.275).warp.AWarpSharp2(depth=-2.5)
    sharp = hvf.LSFmod(unwarp, strength=90, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)

    stabilize = stabilization(out, sharp, 2, 1, 8, 16)
    out = vdf.merge_chroma(stabilize, denoise)



    antialias = lvf.sraa(out, 1.4, rep=7, downscaler=core.resize.Bicubic)
    out = lvf.rfs(antialias, vdf.merge_chroma(upscale, denoise), [(edstart, edend)])



    deband_mask = detail_mask_func(out, brz_a=2000, brz_b=1000)
    deband = dbs.f3kpf(out, 17, 42, 36)

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
                     [(opstart+425, opstart+698), (36134, src.num_frames-1)])
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, src, creditless_mask, 0),
                     [(opstart, opend), (edstart, edend)])
    credit = lvf.rfs(credit, src, [(14566, 14637)])
    out = credit


    return depth(out, 10)



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
    x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
    x265_cmd += '--frame-threads 8 --pmode --pme --preset slower' + ' '
    x265_cmd += f'--frames {clip.num_frames} --fps 24000/1001 --output-depth 10' + ' '
    x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
    x265_cmd += '--merange 48 --weightb' + ' '
    x265_cmd += '--no-strong-intra-smoothing' + ' '
    x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.0 --no-open-gop --keyint 360 --min-keyint 12 --scenecut 45 --rc-lookahead 120 --bframes 16' + ' '
    x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.85 --qcomp 0.70' + ' '
    x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
    x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<2)} --max-luma {str(235<<2)}'# + ' '

    print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
    process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()

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
                '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
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
