"""OreSuki script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from pathlib import Path
from typing import NamedTuple

import debandshit as dbs
import vardefunc as vdf
import awsmfunc as awf
import havsfunc as hvf

from vsutil import depth, get_y, get_w
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
    chapter = '_assets/chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)


JPBD = infos_bd(r'[BDMV] ORESUKI Are you the only one who loves me\ORESUKI Volume 2\BD_VIDEO\BDMV\STREAM\00001', 0, -26)
JPBD_NCOP = infos_bd(r'[BDMV] ORESUKI Are you the only one who loves me\ORESUKI Volume 1\BD_VIDEO\BDMV\STREAM\00004', 24, -24)
JPBD_NCED = infos_bd(r'[BDMV] ORESUKI Are you the only one who loves me\ORESUKI Volume 1\BD_VIDEO\BDMV\STREAM\00005', 24, -24)
X265 = 'x265'


def warping(clip: vs.VideoNode, strength: float, awarp_depth: float)-> vs.VideoNode:
    darken = hvf.Toon(clip, strength)

    darken_mask = core.std.Expr(
        [core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
         core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (clip.format.bits_per_sample-1),
                                                        peak=(1 << clip.format.bits_per_sample)-1)])

    unwarp = core.warp.AWarpSharp2(darken, depth=awarp_depth)

    return core.std.MaskedMerge(clip, unwarp, darken_mask)



def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src
    if out.num_frames < 34046:
        while out.num_frames != 34046:
            out += out[-1]
    opstart, opend = 0, 2157
    edstart, edend = 31768, 33925
    h = 720
    w = get_w(h)


    fixedges = awf.bbmod(out, 2, 2, 2, 2, 64<<8, 999)
    out = fixedges


    decomb = hvf.Vinverse(out)
    decomb = lvf.rfs(out, decomb, [(2162, 2170)])
    ref = decomb
    out = decomb



    clean = core.knlm.KNLMeansCL(out, h=0.55, a=2, d=3, device_type='gpu', device_id=0, channels='UV')
    clean = core.knlm.KNLMeansCL(clean, h=0.55, a=2, d=3, device_type='gpu', device_id=0, channels='Y')
    diff_den = core.std.MakeDiff(out, clean)
    out = depth(clean, 32)


    luma = get_y(out)
    line_mask = vdf.edge_detect(luma, 'FDOG', 0.05, (1, 1))

    descale = core.descale.Debilinear(luma, w, h)
    upscale = vdf.nnedi3_upscale(descale, correct_shift=False, pscrn=1).resize.Bicubic(src.width, src.height, src_left=.5, src_top=.5)
    rescale = core.std.MaskedMerge(luma, upscale, line_mask)

    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)


    moozzi = warping(out, 0.4, 4)
    sharp = hvf.LSFmod(moozzi, strength=95, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp


    deband_mask = lvf.denoise.detail_mask(out, brz_a=2000, brz_b=1000)
    deband = dbs.f3kpf(out, 17, 30, 30)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband


    grain_org = core.std.MergeDiff(out, diff_den)
    out = grain_org


    credit_mask = vdf.diff_rescale_mask(ref, mthr=40, sw=5, sh=5)
    credit_mask = vdf.region_mask(credit_mask, 10, 10, 10, 10).std.Inflate().std.Inflate()
    antialias = lvf.sraa(ref, 2, 13, downscaler=core.resize.Bicubic)
    credit = lvf.rfs(out, core.std.MaskedMerge(out, antialias, credit_mask),
                     [(2163, 2273), (33926, src.num_frames-1)])
    out = credit


    src_c, ncop, nced = [clip.std.Median() for clip in [src, JPBD_NCOP.src_cut, JPBD_NCED.src_cut]]
    opening_mask = vdf.dcm(out, src_c[opstart:opend+1], ncop[:opend-opstart+1], opstart, opend, 3, 3)
    ending_mask = vdf.dcm(out, src_c[edstart:edend+1], nced[:edend-edstart+1], edstart, edend, 3, 3)
    credit_mask = core.std.Expr([opening_mask, ending_mask], 'x y +').std.Convolution([1]*9)

    credit = lvf.rfs(out, core.std.MaskedMerge(out, src, credit_mask), [(opstart, opend), (edstart, edend)])
    out = credit


    return depth(out, 10)



def sec_to_time(secs):
    hours = secs / 3600
    minutes = (secs % 3600) / 60
    secs = secs % 60
    return "%02d:%02d:%05.4f" % (hours, minutes, secs)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
        x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--frame-threads 8 --pmode --pme --preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.0 --no-open-gop --keyint 240 --min-keyint 24 --scenecut 40 --rc-lookahead 84 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.0 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-2:-2 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    if not os.path.isfile(JPBD.a_src.format(1)):
        print('\n\n\nAudio extraction')
        eac3to_args = ['eac3to', JPBD.src,
                       '2:', JPBD.a_src.format(1),
                       '-log=NUL']
        subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    if not os.path.isfile(JPBD.a_enc_cut.format(1)):
        print('\n\n\nAudio encoding')
        qaac_args = ['--no-delay', '--no-optimize', '--threading',
                     '--start', sec_to_time(JPBD.frame_start / (clip.fps_num/clip.fps_den)),
                     '--end', sec_to_time((JPBD.src_clip.num_frames + JPBD.frame_end) / (clip.fps_num/clip.fps_den))]
        qaac_args_more = ['qaac', JPBD.a_src.format(1), '-V', '127', *qaac_args, '-o', JPBD.a_enc_cut.format(1)]
        subprocess.run(qaac_args_more, text=True, check=True, encoding='utf-8')

    if not os.path.isfile('tags_aac_1.xml'):
        ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
        encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
        f = open("tags_aac.xml", 'w')
        f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                    '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                    '</Tag>', '</Tags>'])
        f.close()

    if not os.path.isfile(JPBD.output_final):
        print('\nFinal muxing')
        mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                    '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                    '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                    '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
        subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


    # Clean up
    files = [JPBD.a_src.format(1), JPBD.a_src.format(2),
             JPBD.a_enc_cut.format(1), JPBD.a_enc_cut.format(2), 'tags_aac.xml']
    for file in files:
        if os.path.exists(file):
            os.remove(file)




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
