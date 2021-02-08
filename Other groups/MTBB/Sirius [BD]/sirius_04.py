"""Sirius script"""
__author__ = 'Vardë'

import sys
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path

from regress import Regress, ReconstructMulti
from adptvgrnMod import sizedgrn
import awsmfunc as awf
import muvsfunc as muvf
import G41Fun as gf
import mvsfunc as mvf
import vardefunc as vdf
import havsfunc as hvf
import xvs

from vsutil import depth, get_y, iterate, get_w, plane, join
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.num_threads = 16

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
    output = name + '.264'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'sirius04', 0, -25)


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
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.8, gamma=1000, nrad=1, mdis=15)

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


def dumb3kdb(clip, radius=16, strength=41):
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

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut

    # Variables
    opstart, opend = 1583, 3740
    edstart, edend = 31647, 33806
    full_zone = [(17022, 17069), (31587, 31646), (33987, src.num_frames-1)] # eyecatch, episode name and next episode
    shabc_zone = [(edstart+15, edstart+1215), (edstart+1882, edstart+2126)]
    h = 720
    w = get_w(h)


    # Bicubic sharp parts don't have bad edges
    edges_a = core.edgefixer.ContinuityFixer(src, *[[2, 1, 1]]*4)
    edges_b = awf.bbmod(src, left=6, thresh=32, blur=200)
    edges = lvf.rfs(edges_a, edges_b, [(edstart+1275, edstart+1757)])
    edges = lvf.rfs(edges, src, [(opstart, opend)] + full_zone)
    out = depth(edges, 32)


    # Denoise
    ref = hvf.SMDegrain(depth(get_y(out), 16), thSAD=450)
    denoise = hybrid_denoise(out, 0.35, 1.4, dict(a=2, d=1), dict(ref=depth(ref, 32)))
    out = denoise




    # Descale
    luma = get_y(out)
    lineart = vdf.edge_detect(luma, 'FDOG', 0.055, (1, 1)).std.Median().std.BoxBlur(0, 1, 1, 1, 1)

    descale_a = core.descale.Despline36(luma, w, h).std.SetFrameProp('descaleKernel', data='spline36')
    descale_b = core.descale.Debicubic(luma, w, h, 0, 1).std.SetFrameProp('descaleKernel', data='sharp_bicubic')
    descale = lvf.rfs(descale_a, descale_b, shabc_zone)




    # Chroma reconstruction
    # y_m is the assumed mangled luma.
    # Descale 1080p -> Bad conversion in 422 720p -> Regular 1080p 420
    radius = 2
    y, u, v = descale, plane(out, 1), plane(out, 2)
    y_m = core.resize.Point(y, 640, 720, src_left=-1).resize.Bicubic(960, 540, filter_param_a=1/3, filter_param_b=1/3)


    # 0.25 for 444 and 0.25 for right shifting
    y_m, u, v = [c.resize.Bicubic(w, h, src_left=0.25 + 0.25, filter_param_a=0, filter_param_b=.5) for c in [y_m, u, v]]


    y_fixup = core.std.MakeDiff(y, y_m)
    yu, yv = Regress(y_m, u, v, radius=radius, eps=1e-7)

    u_fixup = ReconstructMulti(y_fixup, yu, radius=radius)
    u_r = core.std.MergeDiff(u, u_fixup)

    v_fixup = ReconstructMulti(y_fixup, yv, radius=radius)
    v_r = core.std.MergeDiff(v, v_fixup)


    # -0.5 * 720/1080 = -1/3
    # -1/3 for the right shift
    # https://forum.doom9.org/showthread.php?p=1802716#post1802716
    u_r, v_r = [c.resize.Bicubic(960, 540, src_left=-1/3, filter_param_a=-.5, filter_param_b=.25) for c in [u_r, v_r]]


    upscale = vdf.fsrcnnx_upscale(descale, height=h*2, shader_file=r'shaders\FSRCNNX_x2_56-16-4-1.glsl',
                                  upscaler_smooth=eedi3_upscale, profile='zastin')

    antialias = sraa_eedi3(upscale, 3, alpha=0.2, beta=0.4, gamma=40, nrad=3, mdis=20)

    downscale = muvf.SSIM_downsample(antialias, src.width, src.height, filter_param_a=0, filter_param_b=0)
    downscale = core.std.MaskedMerge(luma, downscale, lineart)

    merged_a = join([downscale, u_r, v_r])
    merged_b = vdf.merge_chroma(downscale, denoise)
    merged = lvf.rfs(merged_a, merged_b, shabc_zone +
                     [(0, 1582), (opend+1, 3799), (3883, 3978),
                      (4442, 4508), (4802, 5549), (5850, 7777),
                      (8098, 9730), (10040, 10710), (11086, 11533),
                      (11612, 14359), (14948, 15364), (15595, 15916),
                      (15973, 17069), (17129, 17937), (18059, 20208),
                      (20329, 22107), (22360, 22845), (23003, 23619),
                      (23686, 23969), (24183, 24248), (24294, 25125),
                      (25180, 25316), (25371, 25675), (25740, 27388),
                      (27515, 28848), (28921, 31586), (33807, 33986)])
    out = depth(merged, 16)


    warp = xvs.WarpFixChromaBlend(out, 80, 2, depth=8)
    out = warp


    dering = gf.MaskedDHA(out, rx=1.25, ry=1.25, darkstr=0.05, brightstr=1.0, maskpull=48, maskpush=140)
    out = dering


    qtgmc = hvf.QTGMC(out, Preset="Slower", InputType=1, ProgSADMask=2.0)
    qtgmc = vdf.fade_filter(out, out, qtgmc, edstart+1522, edstart+1522+24)
    qtgmc = lvf.rfs(out, qtgmc, [(edstart+1522+25, edstart+1757)])
    out = qtgmc


    out = lvf.rfs(out, depth(denoise, 16), [(opstart, opend)])


    detail_dark_mask = detail_dark_mask_func(get_y(out), brz_a=8000, brz_b=6000)
    detail_light_mask = lvf.denoise.detail_mask(out, brz_a=2500, brz_b=1200)
    detail_mask = core.std.Expr([detail_dark_mask, detail_light_mask], 'x y +').std.Median()
    detail_mask_grow = iterate(detail_mask, core.std.Maximum, 2)
    detail_mask_grow = iterate(detail_mask_grow, core.std.Inflate, 2).std.BoxBlur(0, 1, 1, 1, 1)

    detail_mask = core.std.Expr([get_y(out), detail_mask_grow, detail_mask], f'x {32<<8} < y z ?')



    deband = dumb3kdb(out, 22, 30)
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



    ref = depth(edges, 16)
    credit = out
    rescale_mask = vdf.diff_rescale_mask(ref, h, b=0, c=1, mthr=40, sw=0, sh=0)
    rescale_mask = vdf.region_mask(rescale_mask, *[10]*4)
    rescale_mask = hvf.mt_expand_multi(rescale_mask, mode='ellipse', sw=4, sh=4).std.BoxBlur(0, 1, 1, 1, 1)

    credit = lvf.rfs(credit, ref, full_zone)
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask),
                     [(edstart, edend)])
    out = credit



    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2])
