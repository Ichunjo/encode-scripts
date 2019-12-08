import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w, split, join
core.max_cache_size = 32000

src = lvf.src(r'island03.m2ts')[:34046]
src = fvf.Depth(src, 16)

opstart, opend = 792, 2948
edstart, endend = 31768, 33925

flashback = '[0 472] [13968 14180] [16810 16989] [17134 17217] [28629 28743]'
ed_filter = f'[{edstart} {edstart+1508}] [{edstart+1578} {endend}]'
ed_filter2 = f'[{edstart+1509} {edstart+1578}]'

y, u, v = split(src)

descaled = kgf.inverse_scale(y, height=720, kernel='bicubic', b=1/5, c=1/2, mask_detail=True)
descaled = fvf.Depth(descaled, 16)

adaptmask = vrdf.adaptive_mask(descaled, 300)
edgemask = kgf.retinex_edgemask(descaled, 1)
denoise_a = mvf.BM3D(descaled, 1.2)
denoise_b = mvf.BM3D(descaled, 2.1)
denoise_ed = hvf.SMDegrain(descaled, tr=2, thSAD=150)

denoise = core.std.MaskedMerge(denoise_a, denoise_b, adaptmask)
denoise = core.std.MaskedMerge(denoise, descaled, edgemask)

denoise = fvf.rfs(denoise, denoise_b, mappings=flashback + ed_filter2)
denoise = fvf.rfs(denoise, descaled, mappings='[0 472] [16990 17133]')
denoise = fvf.rfs(denoise, denoise_a, mappings=f'[{opstart} {opstart+106}]')
denoise = fvf.rfs(denoise, denoise_ed, mappings=ed_filter)

aa = lvf.upscaled_sraa(denoise, 2, 13)

scaled_u = core.resize.Bicubic(u, get_w(720), 720, src_left=0.25)
scaled_v = core.resize.Bicubic(v, get_w(720), 720, src_left=0.25)

merged = join([aa, scaled_u, scaled_v])

adaptmask = vrdf.adaptive_mask(merged, 300)
edgemask = kgf.retinex_edgemask(merged, 1)
deband_a = fvf.GradFun3(merged, smode=2, thr=0.15,radius=13)
deband_b = fvf.GradFun3(merged, smode=2, thr=0.35,radius=15)
deband_c = core.f3kdb.Deband(merged, range=15, y=34, cb=26, cr=26, grainy=16, grainc=0, output_depth=16)
deband_c = core.std.MaskedMerge(deband_c, merged, edgemask)
deband_ed = core.f3kdb.Deband(merged, range=12, y=18, cb=12, cr=12, grainy=8, grainc=0, output_depth=16)
deband_ed2 = core.f3kdb.Deband(merged, range=9, y=72, cb=64, cr=64, grainy=32, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_a, deband_b, adaptmask)
deband = fvf.rfs(deband, deband_c, mappings=flashback)
deband = fvf.rfs(deband, deband_a, mappings=f'[{opstart} {opstart+106}]')
deband = fvf.rfs(deband, deband_ed, mappings=ed_filter)
deband = fvf.rfs(deband, deband_ed2, mappings=ed_filter2)

grain_a = kgf.adaptive_grain(deband, 0.3, luma_scaling=16)
grain_b = core.grain.Add(deband, var=0.85, constant=False)
grain_ed = kgf.adaptive_grain(deband, 0.2, luma_scaling=16)
grain = fvf.rfs(grain_a, grain_b, mappings=flashback + f"[{opstart+1624} {opstart+1647}]")
grain = fvf.rfs(grain, grain_ed, mappings=ed_filter)

final = fvf.Depth(grain, 10)

final.set_output()