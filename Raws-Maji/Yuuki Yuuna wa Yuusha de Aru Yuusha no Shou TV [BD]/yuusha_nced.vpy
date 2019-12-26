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
#core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC2/BDMV/STREAM/00004.m2ts')

#asrc = r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC2/BDMV/STREAM/00004.mka'
#ac.eztrim(src,[(0,-24)], asrc,"yuusha_nced.mka")

h = 720

src = src[:-24]

#vrdf.generate_keyframes(src, 'yuusha_nced_keyframes.log')

fixedges = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges32 = fvf.Depth(fixedges, 32)

desc = kgf.inverse_scale(get_y(fixedges32), height=h, kernel='bicubic', b=0, c=1/2, mask_detail=False)

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(fixedges32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa_y = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=2, down8=False)

db_y, db_c = vrdf.F3kdbSep(aa_y, den_c, 15, 30, 30, 20, 0)

grain = kgf.adaptive_grain(db_y, 0.1)

full = vrdf.to444(db_c, grain.width, grain.height, False)

merged = join([grain, full[0], full[1]])

final = fvf.Depth(merged, 10)

#test = vrdf.DRM(src)

final.set_output()