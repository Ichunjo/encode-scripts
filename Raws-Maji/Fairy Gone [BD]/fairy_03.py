"""Fairy Gone script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
from acsuite import eztrim

from adptvgrnMod import sizedgrn
import muvsfunc as muvf
import G41Fun as gf
import mvsfunc as mvf
import vardefunc as vdf
import havsfunc as hvf

from vsutil import depth, get_y, iterate, get_w, scale_value
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
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV] Fairy gone\[BDMV][190717][TBR29111D][Fairy gone フェアリーゴーン Vol.1]\BDROM\BDMV\STREAM\00002', 24, -24)


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

def detail_dark_mask_func(clip: vs.VideoNode, brz_a: int, brz_b: int)-> vs.VideoNode:
    ret = core.retinex.MSRCP(clip, sigma=[100, 250, 800], upper_thr=0.005)
    return lvf.denoise.detail_mask(ret, brz_a=brz_a, brz_b=brz_b)

def dumb3kdbv2(clip, radius=16, strength=41):
    div = (strength - 1) % 16
    if strength < 17:
        return clip
    if div == 0:
        return clip.f3kdb.Deband(radius, strength, strength, strength, 0, 0, output_depth=16)
    lo_str = strength - div
    hi_str = strength + 16 - div
    lo_clip = dumb3kdbv2(clip, radius, lo_str)
    hi_clip = dumb3kdbv2(clip, radius, hi_str)
    return core.std.Merge(lo_clip, hi_clip, (strength - lo_str) / 16)

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut

    # Variables
    opstart, opend = 6161, 8319
    eptitle_s, eptitle_e = 8320, 8439
    edstart, edend = 31170, 33326
    preview_s, preview_e = 33687, src.num_frames-1
    h = 720
    w = get_w(h)

    edges = core.edgefixer.ContinuityFixer(src, *[[2, 1, 1]]*4)
    out = depth(edges, 32)


    ref = hvf.SMDegrain(depth(get_y(out), 16), thSAD=450)
    denoise = hybrid_denoise(out, 0.35, 1.75, dict(a=2, d=1), dict(ref=depth(ref, 32)))
    out = denoise



    y = get_y(out)
    lineart = vdf.edge_detect(y, 'FDOG', 0.055, (1, 1)).std.Median().std.Convolution([*[1]*9])

    descale = core.descale.Debilinear(y, w, h)

    upscale = vdf.fsrcnnx_upscale(descale, height=h*2, shader_file=r'shaders\FSRCNNX_x2_56-16-4-1.glsl',
                                  upscaler_smooth=eedi3_upscale, profile='slow', strength=85)

    antialias = sraa_eedi3(upscale, 9)

    downscale = muvf.SSIM_downsample(antialias, src.width, src.height, filter_param_a=0, filter_param_b=0)
    downscale = core.std.MaskedMerge(y, downscale, lineart)

    merged = vdf.merge_chroma(downscale, out)
    merged = lvf.rfs(merged, out, [(opstart, opend), (eptitle_s, eptitle_e), (preview_s, preview_e)])
    out = depth(merged, 16)




    detail_dark_mask = detail_dark_mask_func(get_y(out), brz_a=8000, brz_b=6000)
    detail_light_mask = lvf.denoise.detail_mask(out, brz_a=2500, brz_b=1200)
    detail_mask = core.std.Expr([detail_dark_mask, detail_light_mask], 'x y +').std.Median()
    detail_mask_grow = iterate(detail_mask, core.std.Maximum, 2)
    detail_mask_grow = iterate(detail_mask_grow, core.std.Inflate, 2).std.Convolution([*[1]*9])

    detail_mask = core.std.Expr([get_y(out), detail_mask_grow, detail_mask], f'x {32<<8} < y z ?')



    deband = dumb3kdbv2(out, 22, 24)
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband





    ref = get_y(out).std.PlaneStats()
    adgmask_a = core.adg.Mask(ref, 30)
    adgmask_b = core.adg.Mask(ref, 12)


    stgrain = sizedgrn(out, 0.1, 0.05, 1.05, sharp=80)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_b)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_a.std.Invert())

    dygrain = sizedgrn(out, 0.2, 0.05, 1.15, sharp=80, static=False)
    dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
    grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
    out = grain





    ref = depth(src, 16)
    rescale_mask = vdf.drm(ref, 720, 'bilinear', mthr=30, sw=0, sh=0)
    rescale_mask = vdf.region_mask(rescale_mask, *[10]*4)
    rescale_mask = hvf.mt_expand_multi(rescale_mask, mode='ellipse', sw=4, sh=4)
    rescale_mask = rescale_mask.std.Binarize(scale_value(100, 8, 16)).std.Inflate().std.Convolution([*[1]*9])

    dehalo_ref = gf.MaskedDHA(ref, rx=1.65, ry=1.65, darkstr=0.15, brightstr=1.0, maskpull=48, maskpush=140)
    dehalo_mask_a = vdf.region_mask(rescale_mask, top=650)
    dehalo_mask_b = vdf.region_mask(rescale_mask, right=400)


    credit = out
    dehalo_range_a = [(25, 205), (518, 612), (2090, 2172), (14449, 14537)]
    dehalo_range_b = [(3893, 3981)]
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask, 0),
                     [(2197, 2252), (4072, 4129), (4409, 4527), (5390, 5484), (8473, 8603),
                      (8610, 8669), (9826, 9909), (10666, 10713), (12088, 12507), (14943, 15052),
                      (17988, 18136), (18897, 18976), (edstart, edend)]
                     + dehalo_range_a + dehalo_range_b)
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, dehalo_ref, dehalo_mask_a, 0),
                     dehalo_range_a)
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, dehalo_ref, dehalo_mask_b, 0),
                     dehalo_range_b)
    out = credit




    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2])



def do_encode(clip):
    """Compression with x26X"""
    vdf.generate_keyframes(JPBD.src_cut, JPBD.name + '_keyframes.txt')

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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.5 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 60 --rc-lookahead 84 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.0 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += f'--qpfile {JPBD.name}_keyframes.txt' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
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
