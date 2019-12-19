import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import debandshit as dbs
from vsutil import insert_clip
from mvsfunc import GetMatrix

import acsuite

core = vs.core
core.max_cache_size = 32000
core.num_threads = 16

ac = acsuite.AC()

src = lvf.src(r'Washio Sumi no Shou Movies/YUYUYU_WASHIO3/BDMV/STREAM/00001.m2ts')

fix = core.imwri.Read(r'Washio Sumi no Shou Movies/YUYUYU_WASHIO3/fixgreenframe.png')
fix = core.std.AssumeFPS(fix, fpsnum=src.fps.numerator, fpsden=src.fps.denominator)
fix = core.resize.Bicubic(fix, src.width, src.height, format=src.format, matrix_s=GetMatrix(src, dIsRGB=True))

asrc = r'Washio Sumi no Shou Movies/YUYUYU_WASHIO3/BDMV/STREAM/00001.mka'
ac.eztrim(src,[(0,-24)], asrc,"washio_sp03.mka")

src = src[:-24]
src = insert_clip(src, fix[:1], 6291)
src = fvf.Depth(src, 16)

aa = lvf.upscaled_sraa(src, sharp_downscale=False)

dering = hvf.HQDeringmod(aa, mrad=1, msmooth=2, mthr=110, drrep=24, thr=13, planes=[0, 1, 2], show=False)

deband_a = dbs.f3kbilateral(dering, range=18, y=50, c=50)
deband_b = dbs.f3kbilateral(dering, range=20, y=70, c=70)
deband = fvf.rfs(deband_a, deband_b, mappings='[6977 7079]')

grain = kgf.adaptive_grain(deband, 0.3, 16)

final = fvf.Depth(grain, 10)
final.set_output()