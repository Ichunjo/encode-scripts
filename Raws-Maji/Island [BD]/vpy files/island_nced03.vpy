import vapoursynth as vs
core = vs.get_core()
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
from vsutil import get_w, split, join
core.max_cache_size = 32000

src = lvf.src(r'island_nced03.m2ts')[:2160]
src = fvf.Depth(src, 16)

edstart, edend = 31768, 33925

y, u, v = split(src)

descaled = kgf.inverse_scale(y, height=720, kernel='bicubic', b=1/5, c=1/2, mask_detail=True)
descaled = fvf.Depth(descaled, 16)

denoise = hvf.SMDegrain(descaled, tr=2, thSAD=150)

aa = lvf.upscaled_sraa(denoise, 2, 13)

scaled_u = core.resize.Bicubic(u, get_w(720), 720, src_left=0.25)
scaled_v = core.resize.Bicubic(v, get_w(720), 720, src_left=0.25)

merged = join([aa, scaled_u, scaled_v])

deband = core.f3kdb.Deband(merged, range=12, y=18, cb=12, cr=12, grainy=8, grainc=0, output_depth=16)

grain = kgf.adaptive_grain(deband, 0.2, luma_scaling=16)

final = fvf.Depth(grain, 10)

final.set_output()