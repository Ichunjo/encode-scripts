"""Lapis script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from functools import partial
from pathlib import Path

from adptvgrnMod import adptvgrnMod
import vardefunc as vdf
import muvsfunc as muvf
import mvsfunc as mvf
import G41Fun as gf
import rekt as rkt

from vsutil import depth, get_y, iterate, get_w
import vapoursynth as vs
import lvsfunc as lvf

core = vs.core
# core.num_threads = 12


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
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][201125][GNXA-2292][Lapis_Re_LiGHTs][vol.2]\LAPIS_RE_LIGHTS_2\BDMV\STREAM\00010', 0+0*(240+12), 240+0*(240+12))


def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                   knlm_args: Optional[Dict[str, Any]] = None,
                   bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
    """Denoise luma with BM3D and chroma with knlmeansCL

    Args:
        clip (vs.VideoNode): Source clip.
        knlm_h (float, optional): h parameter in knlm.KNLMeansCL. Defaults to 0.5.
        sigma (float, optional): Sigma parameter in mvf.BM3D. Defaults to 2.
        knlm_args (Optional[Dict[str, Any]], optional): Optional extra arguments for knlm.KNLMeansCL. Defaults to None.
        bm3d_args (Optional[Dict[str, Any]], optional): Optional extra arguments for mvf.BM3D. Defaults to None.

    Returns:
        vs.VideoNode: Denoised clip
    """
    knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
    if knlm_args is not None:
        knargs.update(knlm_args)

    b3args = dict(radius1=1, profile1='fast')
    if bm3d_args is not None:
        b3args.update(bm3d_args)

    luma = get_y(clip)
    luma = mvf.BM3D(luma, sigma, **b3args)
    chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

    return vdf.merge_chroma(luma, chroma)

def eedi3_upscale(clip: vs.VideoNode, correct_shift: bool = True)-> vs.VideoNode:
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, etype=1, pscrn=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.25, gamma=1000, nrad=2, mdis=20)

    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)
    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)

    return core.resize.Bicubic(clip, src_top=.5, src_left=.5) if correct_shift else clip

def sraa_eedi3(clip: vs.VideoNode, rep: Optional[int] = None, **eedi3_args: Any)-> vs.VideoNode:
    """Drop half the field with eedi3+nnedi3 and interpolate them.

    Args:
        clip (vs.VideoNode): Source clip.
        rep (Optional[int], optional): Repair mode. Defaults to None.

    Returns:
        vs.VideoNode: AA'd clip
    """
    nnargs: Dict[str, Any] = dict(nsize=0, nns=3, qual=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    eeargs.update(eedi3_args)

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3, core.nnedi3cl.NNEDI3CL

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgsf.Repair(flt, clip, rep)

    return flt




def do_filter():
    """Vapoursynth filtering"""
    # Source and dithering
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src



    # Variables
    h = 846
    w = get_w(h)
    cubic_filters = ['catrom', 'mitchell', 'robidoux', 'robidoux sharp']
    cubic_filters = [vdf.get_bicubic_params(cf) for cf in cubic_filters]



    # Remove the dynamic grain
    degrain = hybrid_denoise(out, 0.35, 1.2, dict(a=2, d=1))
    out = degrain



    y = get_y(out)
    y32 = depth(y, 32)
    lineart = vdf.edge_detect(y32, 'kirsch', 0.055, (1, 1)).std.Median().std.Inflate()



    # Use multiple descaling kernel for a sharper result
    descale_clips = [core.descale.Debicubic(y32, w, h, b, c) for b, c in cubic_filters]
    descale = core.std.Expr(descale_clips, 'x y z a min min min x y z max max min')

    # Fix descaling artifacts (yes even for catrom there's still artifacts)
    conv = core.std.Convolution(descale, [1, 2, 1, 2, 0, 2, 1, 2, 1])
    thr, coef = 0.013, 3.2
    descale_fix = core.std.Expr([descale, conv], f'x y - abs {thr} < y x ?').std.PlaneStats()
    adapt_mask = core.adg.Mask(descale_fix, 12).std.Invert().std.Expr(f'x 0.80 - {coef} * 0.20 + 0 max 1 min')

    descale = core.std.MaskedMerge(descale, descale_fix, adapt_mask)



    # Double using eedi3+nnedi, fsrcnnx and a sharpener
    upscale = vdf.fsrcnnx_upscale(descale, w*2, h*2, r'shaders\FSRCNNX_x2_56-16-4-1.glsl', upscaler_smooth=eedi3_upscale,
                                  profile='zastin', sharpener=partial(gf.DetailSharpen, sstr=1.25, power=4))

    # Antialiasing by eedi3
    aa_strong = sraa_eedi3(upscale, 13, alpha=0.3, beta=0.5, gamma=40)
    aa = aa_strong

    # Rescale to 1080p with Bicubic b=0, c=0 AKA Hermite
    down = muvf.SSIM_downsample(aa, src.width, src.height, filter_param_a=0, filter_param_b=0)

    upscale = depth(
        core.std.MaskedMerge(y32, down, lineart), 16
    )

    merged = vdf.merge_chroma(upscale, out)
    out = merged



    # Deband with prefilter
    y = get_y(out)
    detail_light_mask = lvf.denoise.detail_mask(y, brz_a=2500, brz_b=1200)


    pf = iterate(out, core.std.Maximum, 2).std.Convolution([10] * 9, planes=0)
    diff = core.std.MakeDiff(out, pf)

    deband = core.f3kdb.Deband(pf, 17, 36, 36, 36, 12, 12, 2, keep_tv_range=True, output_depth=16)
    deband = core.std.MergeDiff(deband, diff)
    deband = core.std.MaskedMerge(deband, out, detail_light_mask)
    out = deband



    # Regraining
    grain = adptvgrnMod(out, 0.25, 0.15, size=out.height/h, sharp=80, luma_scaling=10, static=True)
    out = grain



    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2])


def sec_to_time(secs):
    hours = secs / 3600
    minutes = (secs % 3600) / 60
    secs = secs % 60
    return "%02d:%02d:%05.4f" % (hours, minutes, secs)

def do_encode(clip):
    """Compression with x26X"""
    # vdf.generate_keyframes(JPBD.src_cut, JPBD.name + '_keyframes.txt')

    if not os.path.isfile(JPBD.output):
        # print('\n\n\nVideo encoding')
        # bits = clip.format.bits_per_sample
        # x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
        # x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
        # x265_cmd += '--preset veryslow' + ' '
        # x265_cmd += f'--frames {clip.num_frames} --fps 24000/1001 --output-depth 10' + ' '
        # x265_cmd += '--high-tier --ref 6' + ' '
        # x265_cmd += '--rd 6 --ctu 64 --min-cu-size 8 --limit-refs 0 --no-limit-modes --rect --amp --no-early-skip --rskip 0 --tu-intra-depth 4 --tu-inter-depth 4 --rd-refine --rdoq-level 2 --limit-tu 0' + ' '
        # x265_cmd += '--max-merge 5 --me star --subme 7 --merange 57 --weightb' + ' '
        # x265_cmd += '--no-strong-intra-smoothing' + ' '
        # x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.5 --no-open-gop --keyint 360 --min-keyint 24 --scenecut 45 --rc-lookahead 120 --b-adapt 2 --bframes 16' + ' '
        # x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.0 --cutree --qcomp 0.70' + ' '
        # x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
        # x265_cmd += '--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma 64 --max-luma 940' + ' '

        # print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        # process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        # clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
        #             print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        # process.communicate()
        ffv1_args = [
            'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
            '-g', '1', '-level', '3', '-threads', '8',
            '-slices', '24', '-slicecrc', '1', JPBD.name + "_lossless.mkv"
        ]
        print("Encoder command: ", " ".join(ffv1_args), "\n")
        process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()


    # print('\n\n\nAudio extraction')
    # eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
    # subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    # qaac_args = ['--no-delay', '--no-optimize', '--threading', '--ignorelength',
    #              '--start', sec_to_time(JPBD.frame_start / (clip.fps_num/clip.fps_den)),
    #              '--end', sec_to_time(JPBD.frame_end / (clip.fps_num/clip.fps_den))]
    # qaac_args_more = ['qaac', JPBD.a_src.format(1), '-V', '127', *qaac_args, '-o', JPBD.a_enc_cut.format(1)]
    # subprocess.run(qaac_args_more, text=True, check=True, encoding='utf-8')


    # ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
    # encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    # f = open("tags_aac.xml", 'w')
    # f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
    #               '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
    #               '</Tag>', '</Tags>'])
    # f.close()

    # print('\nFinal muxing')
    # mkv_args = ['mkvmerge', '-o', JPBD.output_final,
    #             '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
    #             '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
    #             '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
    # subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # # Clean up
    # files = [JPBD.a_src, JPBD.a_src_cut.format(1),
    #          JPBD.a_enc_cut.format(1), 'tags_aac.xml']
    # for file in files:
    #     if os.path.exists(file):
    #         os.remove(file)



if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
