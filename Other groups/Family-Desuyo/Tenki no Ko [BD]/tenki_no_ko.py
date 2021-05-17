"""Tenki no KO script"""
__author__ = 'Vardë'

import sys
import subprocess
from typing import NamedTuple
from pathlib import Path

from cooldegrain import CoolDegrain
import debandshit as dbs
import vardefunc as vdf
import kagefunc as kgf

from vsutil import depth
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
    src = path + 'tenki_no_ko_bluray_raw.264'
    src_clip = core.lsmas.LWLibavSource(src)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + 'tenki_no_ko_{}.w64'
    a_src_cut = path + 'tenki_no_ko_cut_track_{}.w64'
    a_enc_cut = path + 'tenki_no_ko_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.264'
    chapter = path + 'chapters_cut.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r"WEATHERING_WITH_YOU Collector's Edition\WEATHERING_WITH_YOU MOVIE\BDMV\STREAM\\", 552, -24)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)

    src = core.std.FreezeFrames(src, 45219, 45219, 45219-1)


    adg_mask = core.adg.Mask(src.std.PlaneStats(), 20)
    denoise_a = CoolDegrain(src, tr=1, thsad=48, blksize=8, overlap=4, plane=4)
    denoise_b = CoolDegrain(src, tr=1, thsad=24, blksize=8, overlap=4, plane=4)
    denoise = core.std.MaskedMerge(denoise_a, denoise_b, adg_mask)
    denoise = vdf.merge_chroma(denoise, denoise_a)
    out = denoise




    deband_mask = lvf.denoise.detail_mask(out.std.Median(), brz_a=3000, brz_b=1500)
    deband = dbs.f3kpf(out, 20, 24, 30)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband



    grain = kgf.adaptive_grain(out, 0.1, luma_scaling=16)
    out = grain


    return depth(out, 10)


X264_ARGS = dict(
    threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-2', me='tesa', subme=11, psy_rd='1.0:0.00', merange=32,
    keyint=240, min_keyint=23, rc_lookahead=144, crf=14, qcomp=0.7, aq_mode=3, aq_strength=0.90
)

def sec_to_time(secs):
    hours = secs / 3600
    minutes = (secs % 3600) / 60
    secs = secs % 60
    return "%02d:%02d:%05.4f" % (hours, minutes, secs)

def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    vdf.encode(clip, 'x264', JPBD.output, **X264_ARGS)

    print('\n\n\nAudio cutting & encoding')
    qaac_args = ['-V', '127', '--no-delay', '--no-optimize', '--ignorelength', '--threading',
                 '--start', sec_to_time(JPBD.frame_start / (clip.fps_num/clip.fps_den)),
                 '--end', sec_to_time((JPBD.src_clip.num_frames + JPBD.frame_end) / (clip.fps_num/clip.fps_den))]
    for i in range(1, 2+1):
        qaac_args_more = ['qaac', JPBD.a_src.format(i), *qaac_args, '-o', JPBD.a_enc_cut.format(i)]
        subprocess.run(qaac_args_more, text=True, check=True, encoding='utf-8')


    ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder',
                    '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
    encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    fil = open("tags_aac.xml", 'w')
    fil.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                    '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                    '</Tag>', '</Tags>'])
    fil.close()

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--track-name', '0:AVC BDRip par Vardë@Family-Desuyo', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0 Musiques japonaises', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 5.1 Musiques japonaises', '--language', '0:jpn', JPBD.a_enc_cut.format(2),
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0 Musiques anglaises', '--language', '0:jpn', 'tenki_no_ko_eng_song_2.0.m4a',
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 5.1 Musiques anglaises', '--language', '0:jpn', 'tenki_no_ko_eng_song_5.1.m4a',
                '--chapter-language', 'fr', '--chapters', JPBD.chapter]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
