"""Edens Zero script"""
__author__ = 'Vardë'

import os
import shlex
import subprocess
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
import vardefunc as vdf
import havsfunc as hvf
import fvsfunc as fvf
import lvsfunc as lvf

from init_source import Infos

from vsutil import depth, get_y
import vapoursynth as vs

core = vs.core


NUM = __file__[-5:-3]
WEB = Infos()
WEB.set_infos(fr'{NUM}\Edens Zero - {NUM} (Netflix 1080p).mkv', 24, -24)

WEB.a_src = str(WEB.path) + '_audio.aac'
WEB.a_src_cut = str(WEB.path) + '_audio_cut.aac'



def do_filter():
    """Vapoursynth filtering"""
    src = WEB.src_cut
    src = src.std.AssumeFPS(src)


    src = depth(src, 16)
    out = src


    deblock = fvf.AutoDeblock(out)
    edgemask = core.std.Sobel(get_y(out))
    out = core.std.MaskedMerge(out, deblock, edgemask)



    denoise = hvf.SMDegrain(out, tr=3, thSAD=300, thSADC=300)
    out = denoise


    dering = hvf.EdgeCleaner(out, 10, smode=1, hot=True)
    out = dering



    y = get_y(out)
    taps = 5
    w, h = 1600, 900
    descale = core.descale.Delanczos(depth(y, 32), w, h, taps)

    upscale = vdf.nnedi3_upscale(descale, pscrn=1)
    rescale = core.resize.Bicubic(upscale, 1920, 1080, filter_param_a=-.5, filter_param_b=.25)
    out = depth(rescale, 16)


    sharp = hvf.LSFmod(out, strength=90, Smode=3, edgemode=0, source=get_y(denoise))
    out = sharp


    merge = vdf.merge_chroma(out, denoise)
    out = merge


    detail_mask = detail_mask_func(out, brz_a=2250, brz_b=1200)
    pref = out.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9)
    deband1 = vdf.dumb3kdb(pref, 17, 45, grain=16, seed=333)
    deband2 = vdf.dumb3kdb(pref, 15, 49, grain=16, sample_mode=4, use_neo=True, blur_first=False, seed=333)
    deband3 = vdf.dumb3kdb(pref, 20, 65, grain=16)

    th_lo, th_hi = 20 << 8, 26 << 8
    strength = '{1} x - {1} {0} - /'.format(th_lo, th_hi)
    deband = core.std.Expr(
        [pref, deband1, deband2],
        [f'x {th_lo} > x {th_hi} < and z ' + strength + ' * y 1 ' + strength +
         f' - * + x {th_lo} <= z y ? ?', 'y'])

    deband = lvf.rfs(deband, deband3, [(0, 254)])
    deband = core.std.MergeDiff(deband, out.std.MakeDiff(pref))
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband



    grain = adptvgrnMod(out, 0.3, 0.15, size=1.2, luma_scaling=16, hi=[128, 240], protect_neutral=False, seed=333)
    out = grain


    ref = denoise
    rescale_mask = vdf.drm(src, 900, 'spline36', mthr=80).std.Maximum()
    credit = out
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask),
                     [(129, 1640), (1925, 2173), (2546, 2652), (2901, 3068),
                      (3194, 3307), (31325, 33541)])
    credit = lvf.rfs(credit, ref, [(33542, src.num_frames - 1)])
    out = credit


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])



def do_encode(clip):
    """Compression with x26X"""
    if not os.path.isfile(WEB.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {WEB.output} - --y4m' + ' '
        x265_cmd += f'--csv {WEB.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        # x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 1.85 --psy-rdoq 1.5 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 40 --rc-lookahead 48 --bframes 16' + ' '
        x265_cmd += '--crf 16 --aq-mode 3 --aq-strength 0.85 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=1:-1 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    mkv_args = ['mkvextract', WEB.src, 'tracks', f'1:{WEB.a_src}']
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(WEB.src_clip, (WEB.frame_start, WEB.frame_end), WEB.a_src, WEB.a_src_cut.format(1))

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', WEB.output_final,
                '--track-name', '0:HEVC WEBRip by Vardë@Natsumi-no-Sekai', '--language', '0:jpn', WEB.output,
                '--track-name', '0:AAC 2.0', '--language', '0:jpn', WEB.a_src_cut.format(1)]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
