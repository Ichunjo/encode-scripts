from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import vapoursynth as vs


core = vs.core
core.max_cache_size = 16384

src = lvf.src(r'Sasami-san@Ganbaranai_ncop.m2ts')

src = src[:-24]

src = fvf.Depth(src, 16)

y = dsc.Descale(src, 1280, 720, kernel='spline36', gray=True)

u, v  = vrdf.to444(src, 1280, 720, False)

descaled = lvf.join([y, u, v])

den = CoolDegrain(descaled, tr=2, thsad=48, blksize=8, overlap=4, plane=4)

aa = TAAmbk(den, aatype='Nnedi3', mtype=1, nsize=2)

line_m = kgf.retinex_edgemask(aa).std.Binarize(9500).std.Inflate()
db = core.neo_f3kdb.Deband(aa, 14, 30, 24, 24, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

final = fvf.Depth(db, 10)

final.set_output()