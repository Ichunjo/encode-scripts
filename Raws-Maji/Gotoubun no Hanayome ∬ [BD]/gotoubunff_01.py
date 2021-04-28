"""Gotoubun FF script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
from functools import partial
from acsuite import eztrim

from adptvgrnMod import sizedgrn
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf
import G41Fun as gf
import mvsfunc as mvf

from vsutil import depth, get_y, insert_clip, scale_value
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
    # src_clip = lvf.src(src, stream_index=0, ff_loglevel=4)
    src_clip = lvf.src(src, stream_index=0)
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

JPBD = infos_bd(r'[BDMV][210317][Gotoubun no Hanayome ∬][Vol.1]\BDMV\STREAM\00005', 0, 34647)


def dumb3kdb(clip: vs.VideoNode, radius=16, strength=41):
    div = (strength - 1) % 16
    if strength < 17:
        return clip
    if div == 0:
        return clip.f3kdb.Deband(radius, strength, strength, strength, 0, 0, output_depth=16)
    lo_str = strength - div
    hi_str = strength - div + 16
    lo_clip = clip.f3kdb.Deband(radius, lo_str, lo_str, lo_str, 0, 0, output_depth=16)
    hi_clip = clip.f3kdb.Deband(radius, hi_str, hi_str, hi_str, 0, 0, output_depth=16)
    return core.std.Merge(lo_clip, hi_clip, (strength - lo_str) / 16)

def line_darkening(clip: vs.VideoNode, strength: float = 0.2, **kwargs)-> vs.VideoNode:
    """Darken lineart through Toon

    Args:
        clip (vs.VideoNode): Source clip.
        strength (float, optional): Strength in Toon. Defaults to 0.2.

    Returns:
        vs.VideoNode: Darked clip.
    """
    darken = hvf.Toon(clip, strength, **kwargs)
    darken_mask = core.std.Expr(
        [core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
         core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (clip.format.bits_per_sample-1),
                                                        peak=(1 << clip.format.bits_per_sample)-1)])
    return core.std.MaskedMerge(clip, darken, darken_mask)

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

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3, core.nnedi3.nnedi3

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgsf.Repair(flt, clip, rep)

    return flt


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src

    opstart, opend = 528, 2685
    edstart, edend = 32370, 34526



    # # Remove the grain
    ref = hvf.SMDegrain(out, tr=1, thSAD=300, plane=4)
    denoise = mvf.BM3D(out, sigma=[1, 0.75], radius1=1, ref=ref)
    out = denoise



    y = get_y(out)
    y32 = depth(y, 32)
    lineart = vdf.edge_detect(y, 'FDOG', scale_value(0.065, 32, 16), (1, 1)).std.Median().std.BoxBlur(0, 1, 1, 1, 1)
    descale = core.descale.Debicubic(y32, 1560, 878, 0, 0.7)
    out = descale



    dering = gf.MaskedDHA(depth(out, 16), rx=1.4, ry=1.4, darkstr=0.15, brightstr=1.0, maskpull=0, maskpush=60)
    out = depth(dering, 32)



    upscale = vdf.eedi3_upscale(out, eedi3_args=dict(alpha=0.1, beta=0.9))
    antialias = sraa_eedi3(upscale, 3, alpha=0.2, beta=0.6, gamma=100, mdis=20, nrad=3)
    downscale = core.resize.Bicubic(antialias, src.width, src.height, filter_param_a=-.5, filter_param_b=.25)
    downscale = depth(downscale, 16)
    out = downscale




    unwarp = line_darkening(out, 0.10).warp.AWarpSharp2(depth=-2.5)
    merged = vdf.merge_chroma(unwarp, denoise)
    motion = gf.MCDegrainSharp(merged, bblur=partial(core.bilateral.Gaussian, sigma=0.65, sigmaV=0),
                               csharp=partial(hvf.LSFmod, strength=60, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True),
                               thSAD=500, rec=True, chroma=True)
    motion = core.std.MaskedMerge(denoise, motion, lineart)
    out = motion




    detail_light_mask = detail_mask_func(out, brz_a=2500, brz_b=1200)

    deband = dumb3kdb(out, 17, 36)
    deband_b = dbs.f3kbilateral(out, 20, 120, 120)
    deband = lvf.rfs(deband, deband_b, [(opstart+1382, opstart+1433)])
    deband = core.std.MaskedMerge(deband, out, detail_light_mask)
    out = deband



    # Refine motion
    pan_a = out[17284:17329]
    delfra = core.std.DeleteFrames(pan_a, [1, 5, 9, 14, 18, 22, 27, 31, 35])
    time = pan_a.num_frames / (out.fps.numerator / out.fps.denominator)
    delfra = core.std.AssumeFPS(delfra, fpsnum=delfra.num_frames * 10000, fpsden=round(time * 10000))
    delfra += delfra[-1]
    newpan_a = gf.JohnFPS(delfra, out.fps.numerator, out.fps.denominator, pel=4)
    motion = insert_clip(out, newpan_a, 17284)
    out = motion





    # Regraining
    ref = get_y(out).std.PlaneStats()
    adgmask_a = core.adg.Mask(ref, 25)
    adgmask_b = core.adg.Mask(ref, 10)

    stgrain_a = core.grain.Add(out, 0.1, 0, seed=333)
    stgrain_a = core.std.MaskedMerge(out, stgrain_a, adgmask_b.std.Invert())

    stgrain_b = sizedgrn(out, 0.2, 0.1, 1.15, sharp=80, seed=333)
    stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_b)
    stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_a.std.Invert())
    stgrain = core.std.MergeDiff(stgrain_b, out.std.MakeDiff(stgrain_a))

    dygrain = sizedgrn(out, 0.3, 0.1, 1.25, sharp=80, static=False, seed=333)
    dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
    grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
    out = grain


    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2], [0, 1, 2])


def do_encode(clip):
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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.25 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 60 --rc-lookahead 60 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.90 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-2:-2 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))

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
                '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
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
