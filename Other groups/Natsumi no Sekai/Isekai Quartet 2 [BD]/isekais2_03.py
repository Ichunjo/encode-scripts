"""Isekai Quartet S2 script"""
__author__ = 'Vardë'

import sys
import os
import shlex
import subprocess
from typing import NamedTuple, Optional
from pathlib import Path
from acsuite import eztrim

import debandshit as dbs
import vardefunc as vdf
import mvsfunc as mvf
import G41Fun as gf

from vsutil import depth, get_w
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
    src_clip = core.lsmas.LWLibavSource(src, prefer_hw=0, ff_loglevel=3)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'Isekai Quartet S2\Vol1\BDROM\BDMV\STREAM\00007', 0, -24)


def endcard_source(file: str, ref: Optional[vs.VideoNode] = None)-> vs.VideoNode:
    if file.endswith(('.jpeg', '.jpg')):
        file = core.ffms2.Source(file)
        vdf.set_ffms2_log_level('warning')
    else:
        file = core.imwri.Read(file)
    return core.std.AssumeFPS(file, ref) if ref else file


def get_h(width: int, aspect_ratio: float = 16 / 9, *, only_even: bool = True)-> int:
    height = width / aspect_ratio
    if only_even:
        return round(height / 2) * 2
    return round(height)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 32)
    edstart, edend = 14969, src.num_frames-1


    denoise = mvf.BM3D(src, 1.1, radius1=1, depth=16)
    out = denoise




    antialias = lvf.sraa(out, 2, 13, downscaler=core.resize.Bicubic, gamma=500, nrad=2, mdis=16)
    out = antialias



    deband_mask = lvf.denoise.detail_mask(out, brz_a=2250, brz_b=1600).std.Median()
    deband = dbs.f3kbilateral(out, 17, 48, 48)
    deband_a = dbs.f3kbilateral(out, 22, 96, 96)
    deband = lvf.rfs(deband, deband_a, [(edstart, edend)])
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband


    grain = core.grain.Add(out, 0.3, constant=True)
    out = grain


    endcard = endcard_source(r'endcards\yande.re 605709 albedo_(overlord) armor cleavage gun horns isekai_quartet maid overlord rem_(re_zero) tagme thighhighs uniform weapon youjo_senki.jpg', src)
    endcard_ar = endcard.width/endcard.height
    endcard_ev = bool(endcard.format.name == 'YUV420P8')

    endcard = depth(endcard, 16)
    endcard = gf.MaskedDHA(endcard, rx=2.0, ry=2.0, darkstr=0.3, brightstr=1.0, maskpull=48, maskpush=140)


    if endcard_ar > 16/9:
        w, h = get_w(src.height, endcard_ar, only_even=endcard_ev), src.height
    elif endcard_ar < 16/9:
        w, h = src.width, get_h(src.width, endcard_ar, only_even=endcard_ev)
    else:
        w, h = src.width, src.height

    endcard = core.resize.Bicubic(endcard, w, h, range_in=1, range=0, dither_type='error_diffusion')
    endcard = lvf.sraa(endcard, 1.45, 7)
    final = core.std.Splice([out, endcard * (17263 - src.num_frames)], mismatch=True)


    return core.resize.Bicubic(final, format=vs.YUV420P10, dither_type='error_diffusion')



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x26X"""
    print('\n\n\nVideo encoding')
    bits = clip.format.bits_per_sample
    x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
    x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
    x265_cmd += '--frame-threads 4 --pmode --pme --preset slower' + ' '
    x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
    x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
    x265_cmd += '--merange 36 --weightb' + ' '
    x265_cmd += '--no-strong-intra-smoothing' + ' '
    x265_cmd += '--psy-rd 1.85 --psy-rdoq 0.8 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 45 --rc-lookahead 60 --bframes 10' + ' '
    x265_cmd += '--crf 16 --aq-mode 3 --aq-strength 0.85 --qcomp 0.75' + ' '
    x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
    x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

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
                '--track-name', '0:HEVC BDRip by Vardë@Natsumi-no-Sekai', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--chapter-language', 'fr', '--chapters', JPBD.chapter]
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
