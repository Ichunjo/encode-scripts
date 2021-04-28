"""Priconne script"""
__author__ = 'Vardë'

import sys
import os
import shlex
import subprocess
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

from regress import Regress, ReconstructMulti
from cooldegrain import CoolDegrain
import debandshit as dbs

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
    src_clip = core.lsmas.LWLibavSource(src, prefer_hw=1, ff_loglevel=3)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.flac'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = '_assets/chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[200212] PRINCESS CONNECT! Re_Dive CHARACTER SONG ALBUM VOL.1\COZX1624\BDMV\STREAM\00004', 19212, 21365)



def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)


    denoise = CoolDegrain(src, tr=1, thsad=24, blksize=8, overlap=4, plane=4)
    out = denoise



    radius = 2
    clip_in = depth(out, 32)
    y, u, v = split(clip_in)
    y_m = core.resize.Bicubic(y, 960, 540, src_left=-.5, filter_param_a=1/3, filter_param_b=1/3)

    def demangle(clip):
        return core.resize.Bicubic(clip, 1920, 1080, src_left=.25)

    y_m, u, v = map(demangle, (y_m, u, v))

    y_fixup = core.std.MakeDiff(y, y_m)
    yu, yv = Regress(y_m, u, v, radius=radius)

    u_fixup = ReconstructMulti(y_fixup, yu, radius=radius)
    u_r = core.std.MergeDiff(u, u_fixup)

    v_fixup = ReconstructMulti(y_fixup, yv, radius=radius)
    v_r = core.std.MergeDiff(v, v_fixup)

    scaled = depth(join([y, u_r, v_r]), 16)
    out = scaled



    deband_mask = detail_mask_func(out, brz_a=2000, brz_b=1000)
    deband = dbs.f3kpf(out, 17, 24, 24)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband

    return depth(out, 10)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
    x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
    x265_cmd += '--frame-threads 8 --pmode --pme --preset slower' + ' '
    x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num/clip.fps_den} --output-depth 10' + ' '
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
    ffmpeg_args = ['ffmpeg', '-i', JPBD.a_src_cut.format(1), '-compression_level', '12', '-lpc_type', 'cholesky', '-lpc_passes', '3', '-exact_rice_parameters', '1', JPBD.a_enc_cut.format(1)]
    subprocess.run(ffmpeg_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                '--track-name', '0:FLAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1)]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # Clean up
    files = [JPBD.a_src, JPBD.a_src_cut.format(1),
             JPBD.a_enc_cut.format(1)]
    for file in files:
        if os.path.exists(file):
            os.remove(file)

if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
