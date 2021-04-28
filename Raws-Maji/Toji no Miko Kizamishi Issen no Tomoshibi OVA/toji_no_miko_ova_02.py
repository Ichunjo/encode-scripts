"""Toji no Miko OVA script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

import kagefunc as kgf
import vardefunc as vdf
import havsfunc as hvf
import mvsfunc as mvf
from fake_rescale import fake_rescale


from vsutil import depth, get_y, scale_value, iterate
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
    # src_clip = lvf.src(src, stream_index=0, ff_loglevel=4)
    src_clip = lvf.src(src, stream_index=0)
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][210224] Toji no Miko Kizamishi Issen no Tomoshibi OVA Vol.1 Fin\TOJI_TOMO_OVA\BDMV\STREAM\00001', 0, -24)
JPBD_NCED = infos_bd(r'[BDMV][210224] Toji no Miko Kizamishi Issen no Tomoshibi OVA Vol.1 Fin\TOJI_TOMO_OVA\BDMV\STREAM\00012', 24, -24)

def dumb3kdb(clip: vs.VideoNode, radius=16, strength=41):
    div = (strength - 1) % 16
    if strength < 17:
        return clip
    if div == 0:
        return clip.f3kdb.Deband(radius, strength, strength, strength, 0, 0, output_depth=16)
    lo_str = strength - div
    hi_str = strength - div + 16
    lo_clip = clip.f3kdb.Deband(radius, lo_str, lo_str, lo_str, 0, 0, output_depth=16)
    hi_clip = clip.f3kdb.Deband(radius, hi_str, hi_str, hi_str, 0, 0, output_depth=16)
    return core.std.Merge(lo_clip, hi_clip, (strength - lo_str) / 16)


def detail_dark_mask_func(clip: vs.VideoNode, brz_a: int, brz_b: int)-> vs.VideoNode:
    ret = core.retinex.MSRCP(clip, sigma=[100, 250, 800], upper_thr=0.005)
    return detail_mask_func(ret, brz_a=brz_a, brz_b=brz_b)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src

    edstart, edend = 31888, src.num_frames-1



    ref = hvf.SMDegrain(out, tr=1, thSAD=300, plane=4)
    denoise = mvf.BM3D(out, [1.5, 1], 1, profile1='lc', ref=ref)
    out = denoise


    y = get_y(out)
    lineart = vdf.edge_detect(y, 'FDOG', scale_value(0.065, 32, 16), (1, 1)).std.Median().std.BoxBlur(0, 1, 1, 1, 1)

    rescale = fake_rescale(y, 837.5, 1/3, 1/3, coef_dering=0.1, coef_dark=1.25,
                           coef_warp=1.15, coef_sharp=0.75, coef_finalsharp=0.85)
    rescale = core.std.MaskedMerge(y, rescale, lineart)

    out = vdf.merge_chroma(rescale, out)





    preden = core.dfttest.DFTTest(get_y(out), ftype=0, sigma=0.5, sbsize=16, sosize=12, tbsize=1)
    detail_dark_mask = detail_dark_mask_func(preden, brz_a=6000, brz_b=5000)
    detail_light_mask = detail_mask_func(preden, brz_a=2500, brz_b=1200)
    detail_mask = core.std.Expr([detail_dark_mask, detail_light_mask], 'x y +').std.Median()
    detail_mask_grow = iterate(detail_mask, core.std.Maximum, 2)
    detail_mask_grow = iterate(detail_mask_grow, core.std.Inflate, 2).std.BoxBlur(0, 1, 1, 1, 1)

    detail_mask = core.std.Expr([preden, detail_mask_grow, detail_mask], f'x {40<<8} < y z ?')

    pf = vdf.merge_chroma(preden, out)
    deband = dumb3kdb(pf, 17, 40)

    deband = core.std.MergeDiff(deband, out.std.MakeDiff(pf))
    deband = core.std.MaskedMerge(deband, out, detail_light_mask)
    out = deband


    grain = kgf.adaptive_grain(out, 0.35)
    out = grain



    ref = depth(src, 16)
    src_c, src_nced = [depth(x, 16) for x in [src, JPBD_NCED.src_cut]]

    ending_mask = vdf.dcm(out, src_c[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 2, 2).std.BoxBlur(0, 2, 2, 2, 2)

    credit = out
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, ending_mask), [(31900, src.num_frames-1)])
    out = credit





    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2], [0, 1, 2])




def do_encode(clip):
    """Compression with x26X"""
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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.5 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 60 --rc-lookahead 60 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.90 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-2:-2 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '3:', JPBD.a_src.format(1), '4:', JPBD.a_src.format(2), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(2), JPBD.a_src_cut.format(2))

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')
    qaac_args = ['qaac', JPBD.a_src_cut.format(2), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(2)]
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
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0 Commentary', '--language', '0:jpn', JPBD.a_enc_cut.format(2),
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
