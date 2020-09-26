"""Rifle is Beautiful script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from pathlib import Path
from typing import NamedTuple

from regress import Regress, ReconstructMulti
import debandshit as dbs
import vardefunc as vdf
import atomchtools as atf
import G41Fun as gf

from vsutil import depth, split, join
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


JPBD = infos_bd(r'[BDMV][Rifle is Beautiful][Blu-Ray BOX BDx4+CDx2 Fin]\RIFLE_IS_BEAUTIFUL_BDBOX1_D1\BDMV\STREAM\00010', 0, 0)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 32)
    out = src


    full_range = core.resize.Bicubic(out, range_in=0, range=1, dither_type='error_diffusion')
    out = full_range


    radius = 3
    y, u, v = split(out)

    y_m = core.resize.Point(y, 960, 1080, src_left=-1)
    y_m = core.resize.Bicubic(y_m, 960, 540)

    def demangle(clip):
        return vdf.nnedi3_upscale(clip, pscrn=0, correct_shift=False).resize.Spline16(src_left=0.5+0.25, src_top=.5)

    y_m, u, v = map(demangle, (y_m, u, v))


    y_fixup = core.std.MakeDiff(y, y_m)
    yu, yv = Regress(y_m, u, v, radius=radius)

    u_fixup = ReconstructMulti(y_fixup, yu, radius=radius)
    u_r = core.std.MergeDiff(u, u_fixup)

    v_fixup = ReconstructMulti(y_fixup, yv, radius=radius)
    v_r = core.std.MergeDiff(v, v_fixup)

    out = join([y, u_r, v_r])
    out = depth(out, 16)



    dehalo = gf.MaskedDHA(out, rx=1.25, ry=1.25, darkstr=0.10, brightstr=1.0, maskpull=46, maskpush=148)
    out = dehalo


    upscale = atf.eedi3Scale(out, 2160, pscrn=0)
    out = upscale

    dehalo = gf.MaskedDHA(out, rx=1.15, ry=1.15, darkstr=0.10, brightstr=1.0, maskpull=46, maskpush=148)
    out = dehalo


    deband_mask = lvf.denoise.detail_mask(out, brz_a=2000, brz_b=1000)
    deband = dbs.f3kpf(out, 28, 48, 48)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband


    grain = core.grain.Add(out, 1)
    out = grain

    return out.std.AssumeFPS(fpsnum=1, fpsden=1)



def sec_to_time(secs):
    hours = secs / 3600
    minutes = (secs % 3600) / 60
    secs = secs % 60
    return "%02d:%02d:%05.4f" % (hours, minutes, secs)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        ffv1_args = [
            'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
            '-g', '1', '-level', '3', '-threads', '8',
            '-slices', '24', '-slicecrc', '1', '-color_range', 'pc', JPBD.name + "_lossless.mkv"
        ]
        print("Encoder command: ", " ".join(ffv1_args), "\n")
        process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
