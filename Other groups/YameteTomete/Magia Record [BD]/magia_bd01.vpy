"""
Magia 01
"""
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
from adptvgrnMod import adptvgrnMod
from G41Fun import Tweak
from acsuite import eztrim
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
import mvsfunc as mvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1, mask: vs.VideoNode = None):
    bits = clip.format.bits_per_sample - 8
    thr = strength * (1 >> bits)

    y = lvf.get_y(clip)

    strong = TAAmbk(y, aatype='Eedi3', alpha=0.25, beta=0.5, gamma=40, nrad=2, 
                    mdis=20, mtype=0, opencl=False)
    weak = TAAmbk(y, aatype='Nnedi3', nsize=3, nns=3, qual=1, mtype=0, opencl=True)
    expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

    aa = core.std.Expr([strong, weak, y], expr)
    merged = core.std.MaskedMerge(y, aa, mask)
    return core.std.ShufflePlanes([merged, clip], [0, 1, 2], vs.YUV)

src = lvf.src('[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00001.m2ts')
asrc = r"[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00001.mka"
blank = r"blank.wav"
eztrim(src, (0, -48), asrc, "magia_bd01a.mka")
eztrim(src, (0, 119), blank, "magia_bd01b.mka")


src = src[:-48]
src = fvf.Depth(src, 16)

den_y = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=0)
den = core.knlm.KNLMeansCL(den_y, a=2, h=0.25, d=3, device_type='gpu', channels='UV')



ref = fvf.Depth(den, 8, 'none')
line_m_a = TAAmbk(ref, mtype=1, showmask=1)
line_m_b = mdf.retinex_edgemask_mod(ref, opencl=True) \
    .std.Binarize(42).rgvs.RemoveGrain(3).rgvs.RemoveGrain(4).std.Inflate()
line_m = core.std.Expr([line_m_a, line_m_b], 'x y +')
line_m = core.std.Expr(line_m, 'x 30 < 0 x 3 * ?').std.Inflate()
line_m = fvf.Depth(line_m, 16, 'none')

aa_a = _nneedi3_clamp(den, mask=line_m)
aa_b = lvf.upscaled_sraa(den, rep=6, sharp_downscale=False)
aa = fvf.rfs(aa_a, aa_b, mappings="[29453 29476] [29510 29532] [29640 29663]"
			                      "[29775 29798] [29866 29889] [30011 30034]")


db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 18, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

grain_a = kgf.adaptive_grain(db, 0.25, luma_scaling=10)
grain_b = adptvgrnMod(db, 1, size=1.5, luma_scaling=0, grain_chroma=False)
grain = fvf.rfs(grain_a, grain_b, mappings="[5149 5598] [8691 10137]")

crop = core.std.Crop(grain, 240, 240, 0, 0)
borders = core.std.AddBorders(crop, 240, 240, 0, 0)
borders = fvf.rfs(grain, borders, mappings="[5149 5598]")

final = fvf.Depth(borders, 10)

"""
#Endcard
"""
endcard = lvf.src('[BDMV][200304][Magia Record][Vol.1]/Scans/endcard1_front.png')
endcard = core.std.AssumeFPS(endcard, fpsnum=src.fps.numerator, fpsden=src.fps.denominator)
endcard = core.std.CropRel(endcard, left=10, top=17, right=17, bottom=23)
endcard = core.resize.Bicubic(endcard, 1920, 1080, format=vs.RGBS, dither_type='error_diffusion')
[endcard := core.w2xc.Waifu2x(endcard, noise=3, scale=1, photo=True) for i in range(5)]
endcard = mvf.ToYUV(endcard, css='420')
endcard = Tweak(endcard, sat=1.2, bright=-0.05, cont=1.2)
"""
"""


final = final + fvf.Depth(endcard, 10)*119

final.set_output()
