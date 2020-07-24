from cooldegrain import CoolDegrainSF

import debandshit as dbs
import vardefunc as vdf
import shinfunc as shf

from vsutil import depth, get_y, get_w
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

w, h, b, c = get_w(792), 792, 0, 1

def _unblur(descale, denoise):
    sangnom = shf.sangnom_upscale(descale, 1080/h, 128)
    sangnom = vdf.merge_chroma(sangnom, denoise)
    sharp = core.warp.AWarpSharp2(depth(sangnom, 16), 128, 1, depth=24)
    return sharp

def _antialias_stonk(descale, aa):
    eeargs = dict(alpha=0.9, beta=0.1, gamma=10, nrad=3, mdis=40)
    nnargs = dict(nsize=3, nns=4, qual=2, pscrn=1)

    upscale = shf.sangnom_upscale(descale, 2, aa)

    aaa = shf.eedi3_upscale(upscale, 1440/upscale.height, eeargs, nnargs)
    aaa = core.sangnom.SangNom(aaa, aa=aa).std.Transpose().sangnom.SangNom(aa=aa).std.Transpose()

    return core.resize.Spline36(aaa, 1920, 1080)

def filtering(clip: vs.VideoNode, edstart: int, edend: int)-> vs.VideoNode:
    """Main function"""

    clip = depth(clip, 32)

    denoise = CoolDegrainSF(clip, tr=1, thsad=36, blksize=8, overlap=4)
    out = denoise


    luma = get_y(out)
    descale = core.descale.Debicubic(luma, w, h, b, c)
    out = descale


    upscale = shf.nnedi3_upscale(out, clip.height/h, dict(nsize=4, nns=4, qual=2, pscrn=1))
    out = upscale


    antialias = _antialias_stonk(descale, 128)
    mask = core.std.Sobel(antialias).std.Maximum().std.Maximum()
    antialias = lvf.rfs(out, core.std.MaskedMerge(out, antialias, mask), [(edstart+683, edstart+730)])
    merged = vdf.merge_chroma(antialias, denoise)
    out = depth(merged, 16)


    sharp = lvf.rfs(out, _unblur(descale, denoise), [(edstart+501, edstart+617)])
    out = sharp


    ref = core.resize.Point(luma, format=vs.YUV420P16)
    credit = core.std.MaskedMerge(out, ref, vdf.drm(ref, h, b=b, c=c), 0)
    out = credit


    deband_a = shf.deband_stonks(out, 24, 8, 1, shf.edge_mask_simple(out, 'prewitt', 2500, (2, 0)))
    deband_b = shf.deband_stonks(out, 31, 16, 3, shf.edge_mask_simple(out, 'prewitt', 2500, (6, 0)))
    deband_c = dbs.f3kpf(out, 17, 36, 36)
    deband = lvf.rfs(out, deband_a, [(edstart+44, edstart+69), (edstart+94, edstart+165),
                                     (edstart+190, edstart+213), (edstart+320, edstart+375),
                                     (edstart+683, edstart+730)])
    deband = lvf.rfs(deband, deband_b, [(edstart+166, edstart+189)])
    deband = lvf.rfs(deband, deband_c, [(edstart+376, edstart+682), (edstart+731, edend)])
    out = deband

    grain = core.neo_f3kdb.Deband(out, preset='depth', grainy=24, grainc=24)
    out = grain

    return out
