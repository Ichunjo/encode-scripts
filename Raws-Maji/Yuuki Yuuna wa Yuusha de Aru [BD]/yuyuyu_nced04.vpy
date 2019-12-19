import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
from vsutil import get_y, join

import acsuite

core = vs.core
core.max_cache_size = 32000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r'[BDMV][アニメ] 結城友奈は勇者である/[BDMV][アニメ][150520] 結城友奈は勇者である 6 (初回限定版BD)/YUYUYU_6/BDMV/STREAM/00003.m2ts')

asrc = r'[BDMV][アニメ] 結城友奈は勇者である/[BDMV][アニメ][150520] 結城友奈は勇者である 6 (初回限定版BD)/YUYUYU_6/BDMV/STREAM/00003.mka'
ac.eztrim(src,[(0,-24)], asrc,"yuyuyu_nced04.mka")

src = src[:-24]

vrdf.generate_keyframes(src, 'yuyuyu_nced04_keyframes.log')

src32 = fvf.Depth(src, 32)

desc = kgf.inverse_scale(get_y(src32), height=720, kernel='bicubic', b=0.3782, c=0.3109, mask_detail=False)  #Robidoux

den = mvf.BM3D(desc, 1, radius1=1, depth=16)
den_c = core.knlm.KNLMeansCL(src32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa_y = taa.TAAmbk(den, aatype='Nnedi3', nsize=5, nns=4, mtype=2)

db_y, db_c = vrdf.F3kdbSep(aa_y, den_c, 15, 36, 36, 24, 0)

full = vrdf.to444(db_c, 1280, 720, False)

merged = join([db_y, full[0], full[1]])

grain = kgf.adaptive_grain(merged, 0.1)

final = fvf.Depth(grain, 10)

final.set_output()