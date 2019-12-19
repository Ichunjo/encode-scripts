import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import debandshit as dbs

import acsuite

core = vs.core
core.max_cache_size = 32000
core.num_threads = 16

ac = acsuite.AC()

src = lvf.src(r'Washio Sumi no Shou Movies/YUYUYU_WASHIO2/BDMV/STREAM/00001.m2ts')

asrc = r'Washio Sumi no Shou Movies/YUYUYU_WASHIO2/BDMV/STREAM/00001.mka'
ac.eztrim(src,[(0,-24)], asrc,"washio_sp02.mka")

src = src[:-24]
src = fvf.Depth(src, 16)

aa = lvf.upscaled_sraa(src, sharp_downscale=False)

dering = hvf.HQDeringmod(aa, mrad=1, msmooth=2, mthr=110, drrep=24, thr=13, planes=[0, 1, 2], show=False)

deband = dbs.f3kbilateral(dering, range=18, y=50, c=50)

grain = kgf.adaptive_grain(deband, 0.3, 16)

final = fvf.Depth(grain, 10)
final.set_output()