"""Yesterday wo Utatte script"""
__author__ = 'Vardë'

import sys
import os
import subprocess
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

import vardefunc as vdf

from vsutil import depth
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
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.264'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

# JPBD = infos_bd(r'[昨日之歌][Yesterday o Utatte][イエスタデイをうたって][BDMV][1080P][DISC×3 Fin][Rev]\SING_YESTERDAY_FOR_ME_DISC3\BDMV\STREAM\00005', 24, -24)
JPBD = infos_bd(sys.argv[1], 24, -24)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut

    if src.num_frames < 2160:
        while src.num_frames != 2160:
            src += src[-1]

    return depth(src, 10)


X264_ARGS = dict(
    threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='tesa', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=240, min_keyint=23, rc_lookahead=72, crf=14, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    vdf.encode(clip, 'x264', JPBD.output, **X264_ARGS, frames=clip.num_frames)

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
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
                '--track-name', '0:AVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1)]
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


JPBD.src_cut.set_output(0)
# FILTERED = do_filter()
# FILTERED.set_output(1)

# FILTERED[0].set_output(1)
# FILTERED[1].set_output(2)
# FILTERED[2].set_output(3)
# FILTERED[3].set_output(4)
