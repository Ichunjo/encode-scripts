"""Mushoku script"""
__author__ = 'Vardë'

import os
import shlex
import subprocess
from functools import partial
from acsuite import eztrim

import vardefunc as vdf
import havsfunc as hvf
import lvsfunc as lvf
import mvsfunc as mvf

import init_source

from vsutil import depth, get_y, split, join, plane
import vapoursynth as vs

core = vs.core



JPBD = init_source.Infos()
JPBD.set_infos(
    r'[BDMV][210421][TBR31094D][『無職転生 ～異世界行ったら本気だす～』 Blu-ray Chapter 1 初回生産限定版]\BDROM\BDMV\STREAM\00000.m2ts', 24, -24, preset='bd'
)



CREDITS = [(0, 131), (204, 267), (337, 400), (611, 682), (800, 864),
           (1233, 1298), (2013, 2259), (31769, 33926)]


def do_filter() -> vs.VideoNode:
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    out = src


    luma = get_y(out)
    rows = [core.std.CropAbs(luma, out.width, 1, top=out.height - 1),
            core.std.CropAbs(luma, out.width, 1, top=out.height - 2)]
    diff = core.std.Expr(rows, 'x y - abs').std.PlaneStats()

    row_fix = vdf.merge_chroma(luma.fb.FillBorders(bottom=1, mode="fillmargins"),
                               out.fb.FillBorders(bottom=2, mode="fillmargins"))

    fixrow = core.std.FrameEval(out, partial(_select_row, clip=out, row_fix=row_fix), prop_src=diff)
    out = fixrow


    out = depth(out, 16)


    # Denoising only the chroma
    pre = hvf.SMDegrain(out, tr=2, thSADC=300, plane=3)
    planes = split(out)
    planes[1], planes[2] = [mvf.BM3D(planes[i], 1.25, radius2=2, pre=plane(pre, i)) for i in range(1, 3)]
    out = join(planes)


    preden = core.dfttest.DFTTest(out, sbsize=16, sosize=12, tbsize=1)
    detail_mask = lvf.mask.detail_mask(preden, brz_a=2500, brz_b=1500)

    deband = vdf.dumb3kdb(preden, 16, threshold=[17, 17], grain=[24, 0])
    deband = core.std.MergeDiff(deband, out.std.MakeDiff(preden))
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband


    decz = vdf.decsiz(out, min_in=128 << 8, max_in=192 << 8)
    out = decz



    ref = depth(src, 16)
    credit = out
    credit = lvf.rfs(out, ref, CREDITS)
    out = credit



    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def _select_row(n, f: vs.VideoFrame, clip: vs.VideoNode, row_fix: vs.VideoNode, thr: float = 1e-2) -> vs.VideoNode:
    return row_fix if f.props['PlaneStatsAverage'] > thr else clip


def do_encode(clip: vs.VideoNode) -> None:
    """Compression with x26X"""
    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
        x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset veryslow' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --rskip 0' + ' '
        x265_cmd += '--subme 5' + ' '
        x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 2.15 --psy-rdoq 2.5 --no-open-gop --keyint 240 --min-keyint 24 --scenecut 40 --bframes 16' + ' '
        x265_cmd += '--crf 15.75 --aq-mode 3 --aq-strength 0.85 --no-cutree --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7' + ' '
        x265_cmd += '--deblock=-2:-2 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += '--zones '
        for i, (cred_s, cred_e) in enumerate(CREDITS):
            x265_cmd += f'{cred_s},{cred_e},b=0.65'
            if i != len(CREDITS) - 1:
                x265_cmd += '/'
        x265_cmd += ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'  # + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    if not os.path.exists(JPBD.a_src.format(1)):
        print('\n\n\nAudio extraction')
        eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
        subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    if not os.path.exists(JPBD.a_src_cut.format(1)):
        print('\n\n\nAudio cutting')
        eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))

    if not os.path.exists(JPBD.a_enc_cut.format(1)):
        print('\n\n\nAudio encoding')
        qaac_args = ['qaac', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
        subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    if not os.path.isfile('tags_aac_1.xml'):
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
                    '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                    '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                    '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
        print(*mkv_args)
        subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


    # Clean up
    files = [JPBD.a_src.format(1), JPBD.a_src_cut.format(1), JPBD.a_enc_cut.format(1),
             'tags_aac_1.xml']
    for file in files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
else:
    JPBD.src_cut.set_output(0)
    FILTERED = do_filter()
    FILTERED.set_output(1)
    # FILTERED[0].set_output(1)
    # FILTERED[1].set_output(2)
    # lvf.comparison.stack_planes(JPBD.src_cut).set_output(0)
