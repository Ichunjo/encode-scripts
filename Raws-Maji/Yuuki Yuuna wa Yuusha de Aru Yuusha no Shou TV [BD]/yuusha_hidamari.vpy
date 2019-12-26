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

src = lvf.src(r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00003.m2ts')

#asrc = r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00003.mka'
#ac.eztrim(src,[(0,-24)], asrc,"yuusha_hidamari.mka")

h = 720

src = src[:-24]

opstart = 2614

#vrdf.generate_keyframes(src, 'yuusha_hidamari_keyframes.log')

src = fvf.Depth(src, 32)

desc = kgf.inverse_scale(get_y(src), height=h, kernel='bicubic', b=0.3782, c=0.3109, mask_detail=True, descale_mask_zones='[2614 4770] [16640 16735] [32861 34812]')
desc_wasuyu = kgf.inverse_scale(get_y(src), height=h, kernel='bicubic', b=0, c=1/2, mask_detail=False)
desc = fvf.rfs(desc, desc_wasuyu, mappings=f'[28441 28791]')

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(src, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa_y = taa.TAAmbk(den, aatype='Nnedi3', nsize=5, nns=4, mtype=2, opencl=True)

db_y_a, db_c_a = vrdf.F3kdbSep(aa_y, den_c, 15, 36, 36, 24, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(aa_y, den_c, 15, 48, 48, 24, 0)
db_y = fvf.rfs(db_y_a, db_y_b, mappings=f'[{opstart} {opstart+49}] [{opstart+1886} {opstart+1911}] [5310 5536]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings=f'[{opstart} {opstart+49}] [{opstart+1886} {opstart+1911}] [5310 5536]')

grain = kgf.adaptive_grain(db_y, 0.1)

full = vrdf.to444(db_c, grain.width, grain.height, False)

merged = join([grain, full[0], full[1]])

final = fvf.Depth(merged, 10)

#test = vrdf.DRM(src)

final.set_output()