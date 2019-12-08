import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
from vsutil import get_w, split, join
core.max_cache_size = 32000

src = lvf.src(r'island_nced01.m2ts')[:2160]
src = fvf.Depth(src, 16)

edstart, endend = 0, 2159

ed_filter = f'[{edstart} {edstart+1508}] [{edstart+1578} {endend}]'
ed_filter2 = f'[{edstart+1509} {edstart+1578}]'

y, u, v = split(src)

descaled = kgf.inverse_scale(y, height=720, kernel='bicubic', b=1/5, c=1/2, mask_detail=True)
descaled = fvf.Depth(descaled, 16)

denoise_a = hvf.SMDegrain(descaled, tr=2, thSAD=150)
denoise_b = mvf.BM3D(descaled, 2.1)

denoise = fvf.rfs(denoise_a, denoise_b, mappings=ed_filter2)

aa = lvf.upscaled_sraa(denoise, 2, 13)

scaled_u = core.resize.Bicubic(u, get_w(720), 720, src_left=0.25)
scaled_v = core.resize.Bicubic(v, get_w(720), 720, src_left=0.25)

merged = join([aa, scaled_u, scaled_v])

deband_a = core.f3kdb.Deband(merged, range=12, y=18, cb=12, cr=12, grainy=8, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(merged, range=9, y=72, cb=64, cr=64, grainy=32, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=ed_filter2)

grain_a = kgf.adaptive_grain(deband, 0.2, luma_scaling=16)
grain_b = kgf.adaptive_grain(deband, 0.3, luma_scaling=16)
grain = fvf.rfs(grain_a, grain_b, mappings=ed_filter2)

final = fvf.Depth(grain, 10)

final.set_output()