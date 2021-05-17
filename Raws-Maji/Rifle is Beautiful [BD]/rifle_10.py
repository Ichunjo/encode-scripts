"""Rifle is Beautiful script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from pathlib import Path
from typing import NamedTuple

import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf
import G41Fun as gf

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


JPBD = infos_bd(r'[BDMV][Rifle is Beautiful][Blu-Ray BOX BDx4+CDx2 Fin]\RIFLE_IS_BEAUTIFUL_BDBOX2_D4\BDMV\STREAM\00002', 0, -26)
JPBD_NCOP = infos_bd(r'[BDMV][Rifle is Beautiful][Blu-Ray BOX BDx4+CDx2 Fin]\RIFLE_IS_BEAUTIFUL_BDBOX1_D1\BDMV\STREAM\00006', 24, -24)
JPBD_NCED = infos_bd(r'[BDMV][Rifle is Beautiful][Blu-Ray BOX BDx4+CDx2 Fin]\RIFLE_IS_BEAUTIFUL_BDBOX1_D1\BDMV\STREAM\00007', 0, -24)
X265 = 'x265'


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src

    opstart, opend = 432, 2589
    edstart, edend = 31768, 33925


    dehalo = gf.MaskedDHA(out, rx=1.35, ry=1.35, darkstr=0.25, brightstr=1.0, maskpull=46, maskpush=148)
    out = dehalo

    antialias = lvf.sraa(out, 1.5, 9, downscaler=core.resize.Bicubic)
    out = antialias



    sharp = hvf.LSFmod(out, strength=75, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp



    deband_mask = lvf.denoise.detail_mask(out, brz_a=2000, brz_b=1000)
    deband = dbs.f3kpf(out, 17, 24, 24)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband



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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.0 --no-open-gop --keyint 360 --min-keyint 12 --scenecut 45 --rc-lookahead 120 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.0 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
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
                    #    '3:', JPBD.a_src.format(2),
                       '-log=NUL']
        subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')


    if not os.path.isfile(JPBD.a_enc_cut.format(1)):
        print('\n\n\nAudio encoding')
        qaac_args = ['--no-delay', '--no-optimize', '--threading',
                     '--start', sec_to_time(JPBD.frame_start / (clip.fps_num/clip.fps_den)),
                     '--end', sec_to_time((JPBD.src_clip.num_frames + JPBD.frame_end) / (clip.fps_num/clip.fps_den))]
        qaac_args_more = ['qaac', JPBD.a_src.format(1), '-V', '127', *qaac_args, '-o', JPBD.a_enc_cut.format(1)]
        subprocess.run(qaac_args_more, text=True, check=True, encoding='utf-8')
        # qaac_args_more = ['qaac', JPBD.a_src.format(2), '-V', '127', *qaac_args, '-o', JPBD.a_enc_cut.format(2)]
        # subprocess.run(qaac_args_more, text=True, check=True, encoding='utf-8')

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
                    # '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0 Commentary', '--language', '0:jpn', JPBD.a_enc_cut.format(2),
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
