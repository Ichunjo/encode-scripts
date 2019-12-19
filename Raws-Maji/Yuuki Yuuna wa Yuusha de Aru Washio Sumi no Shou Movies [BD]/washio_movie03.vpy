import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
from vsutil import get_y, split, join

import acsuite

core = vs.core
core.max_cache_size = 32000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r'Washio Sumi no Shou Movies/YUYUYU_WASHIO3/BDMV/STREAM/00000.m2ts')

asrc = r'Washio Sumi no Shou Movies/YUYUYU_WASHIO3/BDMV/STREAM/00000.mka'
ac.eztrim(src,[(0,-191)], asrc,"washio_movie03.mka")

src = src[:-191]

fixedges = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges32 = fvf.Depth(fixedges, 32)

luma = get_y(fixedges32)

desc = kgf.inverse_scale(luma, height=720, kernel='bicubic', b=0, c=1/2, mask_detail=True, descale_mask_zones='[1726 3885]')

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(fixedges32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=2, down8=False, opencl=True)

db_y_a, db_c_a = vrdf.F3kdbSep(aa, den_c, 15, 30, 30, 20, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(aa, den_c, 16, 48, 48, 20, 0)
db_y = fvf.rfs(db_y_a, db_y_b, mappings='[60953 61196]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings='[60953 61196]')

grain = kgf.adaptive_grain(db_y, 0.1)

down = core.fmtc.resample(luma, 1280, 720, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32, csp=vs.GRAY16)
ed = fvf.rfs(grain, down, mappings='[63886 69408]')

full = vrdf.to444(db_c, 1280, 720, False)

merged = join([ed, full[0], full[1]])

final = fvf.Depth(merged, 10)
final.set_output()