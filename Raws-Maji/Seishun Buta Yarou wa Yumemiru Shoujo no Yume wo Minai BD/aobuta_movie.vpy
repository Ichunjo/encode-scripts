import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
from nnedi3_rpow2CL import nnedi3_rpow2CL
from nnedi3_rpow2 import nnedi3_rpow2
import acsuite
core.max_cache_size = 64000
core.num_threads = 10
ac = acsuite.AC()

# Modified version of fvsfunc which includes a strenght of descale and nnedi3_clamp of lvsfunc 
def DescaleAA(src, w, h, thr=10, kernel='bicubic', b=0, c=1/2, taps=3, desc_str=0.5, expand=3, inflate=3, showmask=False, opencl=False, device=None):

    if kernel.lower().startswith('de'):
        kernel = kernel[2:]

    ow = src.width
    oh = src.height

    bits = src.format.bits_per_sample
    sample_type = src.format.sample_type
    
    if sample_type == vs.INTEGER:
        maxvalue = (1 << bits) - 1
        thr = thr * maxvalue // 0xFF
    else:
        maxvalue = 1
        thr /= (235 - 16)

    # Fix lineart
    src_y = core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)
    deb = fvf.Resize(src_y, w, h, kernel=kernel, a1=b, a2=c, taps=taps, invks=True)
    # Downscale performed with Catrom because I don’t like very much the line darkening of Spline
    #down = core.resize.Spline36(src_y, w, h)
    down = core.resize.Bicubic(src_y, w, h, filter_param_a=0, filter_param_b=1/2)
    desf = core.std.Expr([deb, down], 'x {strength} * y 1 {strength} - * +'.format(strength=desc_str))
    aa = fvf.Depth(nneedi3_clamp(fvf.Depth(desf, 16), alpha=0.3, beta=0.4, sangnom=16), 32)
    if opencl:
        sharp = nnedi3_rpow2CL(aa, nns=4, device=device).resize.Bicubic(ow, oh, filter_param_a=0, filter_param_b=1/2)
    else:
        sharp = nnedi3_rpow2(aa, nns=4).resize.Bicubic(ow, oh, filter_param_a=0, filter_param_b=1/2)
    thrlow = 4 * maxvalue // 0xFF if sample_type == vs.INTEGER else 4 / 0xFF
    thrhigh = 24 * maxvalue // 0xFF if sample_type == vs.INTEGER else 24 / 0xFF
    edgemask = core.std.Prewitt(sharp, planes=0)
    edgemask = core.std.Expr(edgemask, "x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?"
                                       .format(thrhigh=thrhigh, maxvalue=maxvalue, thrlow=thrlow))
    if kernel == "bicubic" and c >= 0.7:
        edgemask = core.std.Maximum(edgemask, planes=0)
    sharp = core.resize.Point(sharp, format=src.format.id)

    # Restore true 1080p
    deb_upscale = fvf.Resize(deb, ow, oh, kernel=kernel, a1=b, a2=c, taps=taps)
    diffmask = core.std.Expr([src_y, deb_upscale], 'x y - abs')
    for _ in range(expand):
        diffmask = core.std.Maximum(diffmask, planes=0)
    for _ in range(inflate):
        diffmask = core.std.Inflate(diffmask, planes=0)

    mask = core.std.Expr([diffmask,edgemask], 'x {thr} >= 0 y ?'.format(thr=thr))
    mask = mask.std.Inflate().std.Deflate()
    out = core.std.MaskedMerge(src, sharp, mask, planes=0)

    if showmask:
        out = mask

    return out

# Modified version of lvsfunc with Eedi3SangNom instead of Eedi3.
# In TAAmbk, defalut aa (sangnom) is 48. I recommand to lower this values since it’s very destructive.
def nneedi3_clamp(clip, strength=1, alpha=0.25, beta=0.5, gamma=40, nrad=2, mdis=20, nsize=3, nns=3, qual=1, sangnom=24):

    bits = clip.format.bits_per_sample - 8
    thr = strength * (1 >> bits)
    strong = taa.TAAmbk(clip, aatype='Eedi3SangNom', alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, aa=sangnom, mtype=0)
    weak = taa.TAAmbk(clip, aatype='Nnedi3', nsize=nsize, nns=nns, qual=qual, mtype=0)
    expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

    if clip.format.num_planes > 1:
        expr = [expr, '']
    aa = core.std.Expr([strong, weak, clip], expr)
    return aa if clip.format.color_family == vs.GRAY else core.std.ShufflePlanes([aa, clip], [0, 1, 2], vs.YUV)


src = lvf.src(r'00006.m2ts')

asrc_a = r"aobuta1.wav"
asrc_b = r"aobuta2.wav"
ac.eztrim(src, [(144,0)], asrc_a, "aobuta1.mka")
ac.eztrim(src, [(144,0)], asrc_b, "aobuta2.mka")

# I don’t want the aniplex introduction
src = src[144:]

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

# getnative give a better result with b=0, c=0.3 ¯\_(ツ)_/¯. The goal here is to avoid ringing with a full descale.
scaled = DescaleAA(src32, 1561, 878, c=0.3, desc_str=0.6)

denoise = mvf.BM3D(scaled, [1.25, 1], radius1=1, depth=16)

deband_a = core.f3kdb.Deband(denoise, range=17, y=36, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(denoise, range=17, y=48, cb=42, cr=42, grainy=24, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings='[49299 49458] [49516 49677] [50055 50183] [75410 76977] [77148 80268]')

grain = kgf.adaptive_grain(deband, 0.25, luma_scaling=6)

# A "deband" or "denoise" to kill the grain in the ending.
ending = fvf.GradFun3(src16, smode=2, thr=0.5, radius=10)
ending = fvf.rfs(grain, ending, mappings='[123504 129287]')

final = fvf.Depth(ending, 10)

final.set_output()