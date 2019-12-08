import vapoursynth as vs
core = vs.get_core()
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
from vsutil import get_w, split, join
core.max_cache_size = 32000

src = lvf.src(r'restage02.m2ts')[:34045]
src = src+src[-1]
src = fvf.Depth(src, 16)

planes = split(src)

descaled = kgf.inverse_scale(planes[0], height=810, kernel='bicubic', b=1/5, c=2/5, mask_detail=True, descale_mask_zones='[2182 4458] [16808 16998] [31768 34045]')
planes[0] = fvf.Depth(descaled, 16)
planes[1], planes[2] = [core.resize.Spline16(plane, get_w(810), 810, src_left=0.25) for plane in planes[1:]]

merged = join(planes)

aa_mask = kgf.retinex_edgemask(merged)
aa = lvf.upscaled_sraa(merged, 2)
aa = core.std.MaskedMerge(merged, aa, aa_mask)

dering = hvf.HQDeringmod(aa, mrad=1, msmooth=2, mthr=110, drrep=24, thr=13, planes=[0, 1, 2])

deband = core.f3kdb.Deband(dering, range=15, y=36, cb=36, cr=36, grainy=32, grainc=0, output_depth=16)

final = fvf.Depth(deband, 10)

final.set_output()