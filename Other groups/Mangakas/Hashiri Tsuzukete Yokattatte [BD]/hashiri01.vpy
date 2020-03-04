import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import mvsfunc as mvf
import debandshit as dbs
import acsuite
from nnedi3_rpow2 import nnedi3_rpow2
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = core.dgdecodenv.DGSource(r'[BDMV][190316] Hashiri Tsuzukete Yokattatte. Vol.1 Fin/HASHIYOKA/BDMV/STREAM/00002.dgi')
asrc = r'[BDMV][190316] Hashiri Tsuzukete Yokattatte. Vol.1 Fin/HASHIYOKA/BDMV/STREAM/00002.mka'
ac.eztrim(src,[(0,21579)], asrc,"hashiri01.mka")

src = src[:21579]

opstart, opend = 1966, 4124
edstart, edend = 19416, 21578

src = fvf.Depth(src, 16)

# Chroma shift
planes = lvf.split(src)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
shift = lvf.join(planes)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(shift, a=2, h=0.65, d=3, device_type='gpu', channels='UV')

# Denoise luma
adapt_m = vrdf.adaptive_mask(planes[0], 30)
den_a = mvf.BM3D(planes[0], 1.5, radius1=1)
den_b = CoolDegrain(planes[0], tr=1, thsad=24, blksize=8, overlap=4, plane=0)
den = core.std.MaskedMerge(den_a, den_b, adapt_m)

# Clip for diff and descale
y_d = core.knlm.KNLMeansCL(den, a=2, h=0.6, d=2, device_type='gpu')
diff_y = core.std.MakeDiff(den, y_d)


y = core.descale.Debicubic(fvf.Depth(y_d, 32), lvf.get_w(864), 864, 0.2620, 0.3690)
y = nnedi3_rpow2(y, 2, nns=4).resize.Spline36(1920, 1080, vs.GRAY16)

scaled = core.std.ShufflePlanes([y, den_uv], [0, 1, 2], vs.YUV)

line_m = kgf.retinex_edgemask(scaled).std.Binarize(9500).std.Inflate()
db_a = core.neo_f3kdb.Deband(scaled, 17, 36, 48, 48, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(dbs.f3kbilateral(scaled, 17, 48, 48), grainy=24, preset='depth')
db = fvf.rfs(db_a, db_b, mappings='[204 977] [1357 1404] [1495 1679]')
db = core.std.MaskedMerge(db, scaled, line_m)


merge = core.std.ShufflePlanes([core.std.MergeDiff(lvf.plane(db, 0), diff_y, 0), db], [0, 1, 2], vs.YUV)
out = merge



# OP ED
clip = core.std.ShufflePlanes([den, src], [0, 1, 2], vs.YUV)
aa = TAAmbk(clip, aatype='Nnedi3', mtype=1)
db_a = core.neo_f3kdb.Deband(aa, 13, 18, 12, 12, 8, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(clip, 13, 18, 12, 12, 8, 0, sample_mode=4)

oped = fvf.rfs(out, db_a, mappings=f'[{opstart} {opend}]')
oped = fvf.rfs(oped, db_b, mappings=f'[{edstart} {edend}]')
out = oped

final = fvf.Depth(out, 10)

final.set_output()