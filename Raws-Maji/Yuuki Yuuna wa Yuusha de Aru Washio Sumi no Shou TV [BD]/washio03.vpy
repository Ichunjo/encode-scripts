import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
from vsutil import get_y, join

import acsuite

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = lvf.src(r'[BDMV][171220][Yuuki Yuuna wa Yuusha de Aru Washio Sumi no Shou][BD-BOX]/YUYUYU_WASHIO_DISC1/BDMV/STREAM/00002.m2ts')

asrc = r'[BDMV][171220][Yuuki Yuuna wa Yuusha de Aru Washio Sumi no Shou][BD-BOX]/YUYUYU_WASHIO_DISC1/BDMV/STREAM/00002.mka'
ac.eztrim(src,[(0,-24)], asrc,"washio03.mka")

h = 720

src = src[:-24]

vrdf.generate_keyframes(src, 'washio03_keyframes.log')

fixedges = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges32 = fvf.Depth(fixedges, 32)

desc = kgf.inverse_scale(get_y(fixedges32), height=h, kernel='bicubic', b=0, c=1/2, mask_detail=True, descale_mask_zones='[4915 7072] [32657 34814]')

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(fixedges32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=2, down8=False, opencl=True)

db_y_a, db_c_a = vrdf.F3kdbSep(aa, den_c, 15, 30, 30, 20, 0)
db_y_b = dbs.f3kbilateral(aa, range=16, y=64)
db_y_b = core.f3kdb.Deband(db_y_b, grainy=20, output_depth=16, preset='depth')
db_c_b = fvf.Depth(dbs.f3kbilateral(den_c, range=16, y=0, c=64), 16)
db_y = fvf.rfs(db_y_a, db_y_b, mappings='[4047 4782]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings='[4047 4782]')

grain = kgf.adaptive_grain(db_y, 0.1)

full = vrdf.to444(db_c, grain.width, grain.height, False)

merged = join([grain, full[0], full[1]])

final = fvf.Depth(merged, 10)

final.set_output()