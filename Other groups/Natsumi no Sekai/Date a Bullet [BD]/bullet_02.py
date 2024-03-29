"""DAB script"""
__author__ = 'Vardë'

import os
import shlex
import subprocess
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
import vardefunc as vdf
import havsfunc as hvf
import mvsfunc as mvf
import G41Fun as gf
import lvsfunc as lvf

from fake_rescale import fake_rescale
from init_source import Infos

from vsutil import depth, get_y, iterate
import vapoursynth as vs

core = vs.core


JPBD = Infos()
JPBD.set_infos(r'[BDMV] Date a Bullet\BDROM\BDMV\STREAM\00005.m2ts', 0, -24)

JPBD_NCED1 = Infos()
JPBD_NCED2 = Infos()
JPBD_NCED1.set_infos(r'[BDMV] Date a Bullet\BDROM\BDMV\STREAM\00007.m2ts', 24, -24)
JPBD_NCED2.set_infos(r'[BDMV] Date a Bullet\BDROM\BDMV\STREAM\00008.m2ts', 24, -24)



def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src

    edstart1, edend1 = 37428, 37428 + 2160
    edstart2, edend2 = edend1 + 12, edend1 + 2160




    ref = hvf.SMDegrain(out, thSAD=300)
    denoise = mvf.BM3D(out, [1.5, 1.25], radius1=1, ref=ref)
    out = denoise




    y = get_y(out)
    lineart = gf.EdgeDetect(y, 'scharr').morpho.Dilate(2, 2).std.Inflate()

    fkrescale = fake_rescale(
        y, 882, 0, 1,
        deringer=lambda x: gf.MaskedDHA(x, rx=1.85, ry=1.85, darkstr=0.25, brightstr=1.0, maskpull=100, maskpush=200),
        antialiser=lambda c: lvf.sraa(c, 2, 13, downscaler=core.resize.Bicubic)
    )
    merged = core.std.MaskedMerge(y, fkrescale, lineart)
    out = vdf.merge_chroma(merged, out)


    dering = hvf.EdgeCleaner(out, 17, smode=1, hot=True)
    out = dering



    out = lvf.rfs(out, denoise, [(edstart1, src.num_frames - 1)])


    detail_mask = lvf.mask.detail_mask(out, brz_a=2250, brz_b=1000)
    deband = vdf.dumb3kdb(out, 15, threshold=17, grain=(24, 0))
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband




    grain = adptvgrnMod(out, 0.3, static=True, grain_chroma=False, hi=[128, 240], seed=333)
    out = grain


    decz = vdf.decsiz(out, min_in=128 << 8, max_in=200 << 8)
    out = decz



    ref = depth(src, 16)
    src_c, src_nced1, src_nced2 = [
        depth(x, 16).std.Convolution(
            [1, 2, 1, 2, 4, 2, 1, 2, 1]
        ) for x in [src, JPBD_NCED1.src_cut, JPBD_NCED2.src_cut]
    ]

    ending_mask1 = vdf.dcm(out, src_c[edstart1:edend1 + 1], src_nced1[:edend1 - edstart1 + 1], edstart1, edend1, 2, 2)
    ending_mask2 = vdf.dcm(out, src_c[edstart2:edend2 + 1], src_nced2[:edend2 - edstart2 + 1], edstart2, edend2, 2, 2)
    ending_mask = core.std.Expr([ending_mask1, ending_mask2], 'x y +')
    ending_mask = iterate(ending_mask, core.std.Inflate, 4)

    credit = out
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, ending_mask), [(edstart1, edend2)])
    credit = lvf.rfs(credit, ref, [(21594, 21737), (39589, 39599), (41744, src.num_frames - 1)])
    out = credit


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def do_encode(clip: vs.VideoNode) -> vs.VideoNode:
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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.5 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 40 --rc-lookahead 60 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.85 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16 << (bits - 8))} --max-luma {str(235 << (bits - 8))}'

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()


    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '3:', JPBD.a_src.format(2), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(2), JPBD.a_src_cut.format(2))


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
