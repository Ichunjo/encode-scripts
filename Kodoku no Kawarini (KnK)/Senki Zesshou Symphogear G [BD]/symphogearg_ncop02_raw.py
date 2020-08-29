"""Symphogear G script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from typing import NamedTuple
from pathlib import Path

import vardefunc as vdf

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
    src_cut = src_clip[frame_start:frame_end]
    name = Path(sys.argv[0]).stem
    output = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, None, None, None,
                   name, output, None, None)


JPBD_NCOP = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][131106] 戦姫絶唱シンフォギアG 2\KIXA_90350\BDMV\STREAM\00007', 24, -24)
JPBD_10 = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][140205] 戦姫絶唱シンフォギアG 5\KIXA_90354\BDMV\STREAM\00004', 0, -24)



def do_filter():
    """Vapoursynth filtering"""
    opstart_ep10 = 768
    ncop = JPBD_NCOP.src_cut
    ep10 = JPBD_10.src_cut

    ncop = lvf.rfs(ncop, ep10[opstart_ep10:], [(0, 79), (1035, 1037)])

    return ncop



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
    if not os.path.exists(JPBD_NCOP.output):
        ffv1_args = [
            'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
            '-g', '1', '-level', '3', '-threads', '8',
            '-slices', '24', '-slicecrc', '1', "_assets/" + JPBD_NCOP.name + "_lossless.mkv"
        ]
        print("Encoder command: ", " ".join(ffv1_args), "\n")
        process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
