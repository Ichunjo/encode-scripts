import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import modfunc as mdf
from vsutil import split, join
core.max_cache_size = 32000

src = lvf.src(r'island_bdmenu01-02.m2ts')[765:766]

src = core.std.AssumeFPS(src, fpsnum=1000, fpsden=307307)
#src = core.std.AssumeFPS(src, fpsnum=1000, fpsden=113113)

denoise = mdf.hybriddenoise_mod(fvf.Depth(src, 32), knl=1.2, sigma=0.25, depth=16)

dering = hvf.HQDeringmod(denoise, mrad=1, msmooth=1, mthr=60, minp=1, sharp=1, drrep=24, thr=12.0, elast=2.0, planes=[0, 1, 2], show=False)

planes = split(dering)
planes[0] = lvf.upscaled_sraa(planes[0], 2)
planes[1], planes[2] = [lvf.upscaled_sraa(plane, 2, 13) for plane in planes[1:]]
aa = join(planes)

deband = core.f3kdb.Deband(aa, range=17, y=30, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)

final = fvf.Depth(deband, 10)

final.set_output()