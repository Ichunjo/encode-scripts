"""Symphogear G script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from typing import NamedTuple
from pathlib import Path

import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf

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


def infos_bd(path, frame_start, frame_end)-> InfosBD:
    src = path + '.mkv'
    src_clip = lvf.src(src); vdf.set_ffms2_log_level('warning')
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = '_chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][131002] 戦姫絶唱シンフォギアG 1\KIXA_90350\BDMV\STREAM\00162', None, None)

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_clip
    src = depth(src, 16)


    cleanedge = hvf.EdgeCleaner(src, 15, rmode=1, smode=1, hot=True)
    out = cleanedge


    deband_mask = detail_mask_func(out, brz_a=2250, brz_b=1500).std.Median()
    deband = dbs.f3kpf(out, 17, 30, 30)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband

    freeze = core.std.FreezeFrames(out, 120, src.num_frames-1, 120)
    out = freeze

    return depth(out, 10)



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
    if not os.path.exists(JPBD.output):
        x265_args = [
            "x265", "--y4m", "--frames", f"{clip.num_frames}", "--sar", "1", "--output-depth", "10",
            "--colormatrix", "bt709", "--colorprim", "bt709", "--transfer", "bt709", "--range", "limited",
            "--min-luma", str(16<<2), "--max-luma", str(235<<2),
            "--fps", f"{clip.fps_num}/{clip.fps_den}",
            "-o", JPBD.output, "-",
            "--frame-threads", "4",
            "--no-sao", "--fades",
            "--preset", "slower",
            "--crf", "15", "--qcomp", "0.70",
            "--bframes", "16",
            "--psy-rd", "2.0", "--psy-rdoq", "1.0",
            "--deblock", "-1:-1",
            "--rc-lookahead", "96",
            "--min-keyint", "23", "--keyint", "360",
            "--aq-mode", "3", "--aq-strength", "1.0"
            ]
        print("Encoder command: ", " ".join(x265_args), "\n")
        process = subprocess.Popen(x265_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    if not os.path.exists(JPBD.a_src):
        print('\n\n\nAudio extraction')
        eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src, '-log=NUL']
        subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    if not os.path.exists(JPBD.a_enc_cut.format(1)):
        print('\n\n\nAudio encoding')
        qaac_args = ['qaac', JPBD.a_src, '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
        subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')


    if not os.path.exists("tags_aac.xml"):
        ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
        encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
        f = open("tags_aac.xml", 'w')
        f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                      '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                      '</Tag>', '</Tags>'])
        f.close()

    if not os.path.exists(JPBD.output_final):
        print('\nFinal muxing')
        mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                    '--timestamps', '0:symphogearg_01_timecode.txt',
                    '--track-name', '0:HEVC BDRip by Vardë@Kodoku-no-Kawarini', '--language', '0:jpn', JPBD.output,
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
