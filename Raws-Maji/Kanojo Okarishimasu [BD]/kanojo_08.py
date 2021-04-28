"""Kanojo Okarishimasu script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf

from vsutil import depth, get_y, iterate
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
    src_clip = lvf.src(src, stream_index=0, ff_loglevel=4)
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = '_chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][201223][Kanojo, Okarishimasu][Vol.03]\BD\BDMV\STREAM\00005', None, None)


def detail_dark_mask_func(clip: vs.VideoNode, brz_a: int, brz_b: int)-> vs.VideoNode:
    ret = core.retinex.MSRCP(clip, sigma=[100, 250, 800], upper_thr=0.005)
    return detail_mask_func(ret, brz_a=brz_a, brz_b=brz_b)



def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    out = depth(src, 16)




    denoise = hvf.SMDegrain(out, thSAD=150, thSADC=75)
    out = denoise




    y = get_y(out)
    lineart = core.std.Sobel(y).std.Binarize(75<<8).std.Maximum().std.Inflate()

    antialias = lvf.sraa(y, 1.5, 9, downscaler=core.resize.Spline36, gamma=200, mdis=18)

    sharp = hvf.LSFmod(antialias, strength=95, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)


    minmax = core.std.Expr([y, sharp, antialias], 'x y z min max y z max min')
    merge = core.std.MaskedMerge(y, minmax, lineart)
    out = vdf.merge_chroma(merge, out)



    y = get_y(out)
    detail_dark_mask = detail_dark_mask_func(y, brz_a=10000, brz_b=9000)
    detail_light_mask = detail_mask_func(y, brz_a=2500, brz_b=1200)
    detail_mask = core.std.Expr([detail_dark_mask, detail_light_mask], 'x y +').std.Median()
    detail_mask_grow = iterate(detail_mask, core.std.Maximum, 2)
    detail_mask_grow = iterate(detail_mask_grow, core.std.Inflate, 2).std.Convolution([1, 1, 1, 1, 1, 1, 1, 1, 1])

    detail_mask = core.std.Expr([y, detail_mask_grow, detail_mask], f'x {28<<8} < y z ?')

    deband = dbs.f3kpf(out, 17, 24, 24)
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband



    grain = adptvgrnMod(out, 0.2, 0.1, 1.25, luma_scaling=14, sharp=80, static=False, lo=19, hi=[192, 240])
    out = grain





    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2], [0, 1, 2])


def do_encode(clip):
    """Compression with x26X"""
    vdf.generate_keyframes(JPBD.src_cut, JPBD.name + '_keyframes.txt')

    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
        x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.25 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 60 --rc-lookahead 84 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.90 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += f'--qpfile {JPBD.name}_keyframes.txt' + ' '
        x265_cmd += '--deblock=-1:0 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()


    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '3:', JPBD.a_src.format(1), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))

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
