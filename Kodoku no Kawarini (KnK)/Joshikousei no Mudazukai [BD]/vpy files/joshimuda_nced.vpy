import vapoursynth as vs
core = vs.get_core()
import muvsfunc as muvf
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_w
core.max_cache_size = 32000

src = lvf.src(r'joshimuda_nced.m2ts')[24:2184]
src = fvf.Depth(src, 16)

luma = core.std.ShufflePlanes(src, 0, vs.GRAY)

descaled = core.descale.Debicubic(fvf.Depth(luma, 32), get_w(864), 864, 0, 1/2)
descaled = fvf.Depth(descaled, 16)

denoise_l = hvf.SMDegrain(descaled, tr=2, thSAD=280, contrasharp=True)

scaled = nnedi3_rpow2CL(denoise_l, device=0).resize.Spline36(1920, 1080)

merged = core.std.ShufflePlanes([scaled, src], [0, 1, 2], vs.YUV)

deband = core.f3kdb.Deband(merged, range=17, y=36, cb=26, cr=26, grainy=0, grainc=0, output_depth=16)
line = kgf.retinex_edgemask(merged)
deband = core.std.MaskedMerge(deband, merged, line)

grain_a = kgf.adaptive_grain(deband, 0.2, luma_scaling=6)
grain_b = kgf.adaptive_grain(grain_a, 0.8, static=False, luma_scaling=200)

final = fvf.Depth(grain_b, 10)
final.set_output()