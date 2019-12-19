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

src = lvf.src(r'Washio Sumi no Shou Movies/YUYUYU_WASHIO1/BDMV/STREAM/00000.m2ts')

asrc = r'Washio Sumi no Shou Movies/YUYUYU_WASHIO1/BDMV/STREAM/00000.mka'
ac.eztrim(src,[(0,68496)], asrc,"washio_movie01.mka")

src = src[:68496]

fixedges = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges32 = fvf.Depth(fixedges, 32)

luma = get_y(fixedges32)

desc = kgf.inverse_scale(luma, height=720, kernel='bicubic', b=0, c=1/2, mask_detail=True, descale_mask_zones='[6343 8502]')

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(fixedges32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=2, down8=False, opencl=True)

db_y, db_c = vrdf.F3kdbSep(aa, den_c, 15, 30, 30, 20, 0)

grain = kgf.adaptive_grain(db_y, 0.1)

down = core.fmtc.resample(luma, 1280, 720, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32, csp=vs.GRAY16)
ed = fvf.rfs(grain, down, mappings='[63817 68135]')

full = vrdf.to444(db_c, 1280, 720, False)

merged = join([ed, full[0], full[1]])

final = fvf.Depth(merged, 10)
final.set_output()