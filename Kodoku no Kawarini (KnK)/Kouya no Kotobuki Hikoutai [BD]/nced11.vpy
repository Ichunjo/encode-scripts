import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
from nnedi3_rpow2 import nnedi3_rpow2
from vsutil import get_y, get_w
import acsuite

core = vs.core
core.max_cache_size = 64000

ac = acsuite.AC()

src = lvf.src(r'Vol.06 Fin/BDMV/STREAM/00011.m2ts')

#asrc = r'Vol.06 Fin/BDMV/STREAM/00011.mka'
#ac.eztrim(src, [(10968,13138)], asrc, "nced11_audios-cut.mka")

src = src[10968:13138]
src = fvf.Depth(src, 16)

fixedges = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])

descaled = core.descale.Debicubic(fvf.Depth(get_y(fixedges), 32), get_w(844), 844, 0, 1/2)

denoise_mask = vrdf.adaptive_mask(descaled, 10)
denoise_a = mvf.BM3D(descaled, sigma=0.35, radius1=1)
denoise_b = mvf.BM3D(descaled, sigma=0.75, radius1=1)
denoise = core.std.MaskedMerge(denoise_b, denoise_a, denoise_mask)
denoise = fvf.Depth(denoise, 16)

db_y, db_c = vrdf.F3kdbSep(denoise, src, 16, 36, 24, 0, 0)

aa = taa.TAAmbk(db_y, aatype='Eedi3', alpha=0.2, beta=0.25, gamma=1000, nrad=2, mdis=15, mtype=1, mpand=(2,2), sharp=60)

scaled = nnedi3_rpow2(aa, nns=4).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=0)

merged = core.std.ShufflePlanes([scaled, db_c], [0, 1, 2], vs.YUV)

grain = kgf.adaptive_grain(merged, 0.2, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()