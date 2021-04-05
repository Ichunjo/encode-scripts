"""Kouya Movie script"""
__author__ = 'Vardë'

import sys
import subprocess
from functools import partial
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path

from adptvgrnMod import sizedgrn
import vardefunc as vdf
import muvsfunc as muvf
import havsfunc as hvf
import mvsfunc as mvf
import G41Fun as gf

from vsutil import depth, get_y, get_w
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
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'Kouya no Kotobuki Hikoutai Kanzenban JP BDMV\KOTOBUKI_THE_MOVIE\BDMV\STREAM\00002', None, None)
JPBD_NCED = infos_bd(r'Kouya no Kotobuki Hikoutai Kanzenban JP BDMV\KOTOBUKI_THE_MOVIE\BDMV\STREAM\00004', 48, None)


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
    src = src.std.SetFrameProp('_Matrix', intval=1)

    # Variables
    edstart, edend = 165528, src.num_frames-1
    h = 844
    w = get_w(h)


    edges_a = core.edgefixer.ContinuityFixer(src, *[[1, 0, 0]]*4)
    edges_b = core.edgefixer.ContinuityFixer(src, [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0])
    edges = lvf.rfs(edges_a, edges_b, [(8900, 9001)])
    out = depth(edges, 32)


    ref = hvf.SMDegrain(depth(get_y(out), 16), thSAD=300)
    denoise = hybrid_denoise(out, 0.25, 1.05, dict(a=2, d=1), dict(ref=depth(ref, 32)))
    out = denoise



    y = get_y(out)
    lineart = vdf.edge_detect(y, 'FDOG', 0.065, (1, 1)).std.Median().std.BoxBlur(0, 1, 1, 1, 1)


    descale_clips = [core.resize.Bicubic(y, w, h, filter_param_a=1/3, filter_param_b=1/3),
                     core.descale.Debicubic(y, w, h, 0, 1/2),
                     core.descale.Debicubic(y, w, h, 1/3, 1/3)]
    descale = core.std.Expr(descale_clips, 'x y z min max y z max min z min')
    descale_bcsharp = core.descale.Debicubic(y, get_w(842), 842, 0, 1)
    descale_bcsharp = gf.MaskedDHA(depth(descale_bcsharp, 16), rx=1.4, ry=1.4, darkstr=0.05, brightstr=1.0, maskpull=48, maskpush=140)


    upsc_args = dict(shader_file='shaders/FSRCNNX_x2_56-16-4-1.glsl', upscaler_smooth=eedi3_upscale, profile='zastin',
                     sharpener=partial(gf.DetailSharpen, sstr=1.65, power=4, mode=0, med=True))

    upscale = vdf.fsrcnnx_upscale(descale, height=h*2, **upsc_args)
    antialias = sraa_eedi3(upscale, 3, alpha=0.2, beta=0.4, gamma=100, mdis=20, nrad=3)
    downscale_a = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic', filter_param_a=0, filter_param_b=0)


    upscale = eedi3_upscale(descale_bcsharp)
    antialias = sraa_eedi3(depth(upscale, 32), 3, alpha=0.2, beta=0.4, gamma=100, mdis=20, nrad=3)
    downscale_b = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic', filter_param_a=0, filter_param_b=0)



    downscale = lvf.rfs(downscale_a, downscale_b, [(52931, 53004), (53459, 53696), (54218, 54441),
                                                   (55137, 55254), (55420, 55570)])
    scaled = core.std.MaskedMerge(y, downscale, lineart)

    merged = vdf.merge_chroma(scaled, out)
    out = depth(merged, 16)



    crop = core.std.Crop(out, left=4, right=2)
    fb = core.fb.FillBorders(crop, 1, 1, 0, 0, mode="fillmargins")
    resize = core.resize.Bicubic(fb, 1920)
    out = lvf.rfs(out, resize, [(8900, 9001)])



    detail_light_mask = lvf.denoise.detail_mask(out, brz_a=2500, brz_b=1200)

    deband = dumb3kdb(out, 16, 42)
    deband = core.std.MaskedMerge(deband, out, detail_light_mask)
    out = deband





    ref = get_y(out).std.PlaneStats()
    adgmask_a = core.adg.Mask(ref, 30)
    adgmask_b = core.adg.Mask(ref, 12)

    stgrain = sizedgrn(out, 0.1, 0.05, 1.05, sharp=80, static=True)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_b)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_a.std.Invert())

    dygrain = sizedgrn(out, 0.2, 0.05, 1.15, sharp=80, static=False)
    dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
    grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
    out = grain




    ref = depth(src, 16)
    src_c, src_nced = [c.std.BoxBlur(0, 1, 1, 1, 1) for c in [src, JPBD_NCED.src_cut]]
    creditless_mask = vdf.dcm(ref, src_c[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 3, 3).std.Deflate()
    rescale_mask = vdf.drm(ref, h, b=1/3, c=1/3, mthr=30)
    rescale_mask = vdf.region_mask(rescale_mask, 20, 20, 20, 20).std.Binarize(6000).std.Maximum().std.Maximum().std.Inflate()

    credit = out
    credit = lvf.rfs(credit, ref, [(0, 672)])
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, creditless_mask, 0), [(edstart, edend)])
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask[5602], 0), [(5561, 5752)])
    out = credit


    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2])




def do_encode(clip):
    """Compression with x26X"""
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




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
