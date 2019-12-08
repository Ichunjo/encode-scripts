import vapoursynth as vs
core = vs.get_core()
import muvsfunc as muvf
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import *
core.max_cache_size = 26000

src = lvf.src(r'asuka_nced.m2ts')[24:2184]
src = fvf.Depth(src, 16)

edstart = 0

scaled = fvf.DescaleAA(src, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)

denoise_a = vrdf.KNLMeansCL(scaled, 0.55, 0, 0)
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise_a = core.std.MaskedMerge(denoise_a, scaled, denoise_mask)
denoise_b = vrdf.KNLMeansCL(scaled, 0.75, 0.45, 0)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')

line = kgf.retinex_edgemask(denoise)

aa_a = lvf.transpose_aa(denoise, eedi3=True)
aa_a = core.std.MaskedMerge(denoise, aa_a, line)
aa_c = lvf.upscaled_sraa(get_y(denoise))
aa_c = core.std.ShufflePlanes([aa_c, denoise], [0, 1, 2], vs.YUV)
aa = fvf.rfs(aa_a, aa_c, mappings=f'[{edstart+66} {edstart+125}]')

deband_a = core.f3kdb.Deband(aa, range=17, y=34, cb=26, cr=26, grainy=16, grainc=0, output_depth=16)
deband_a = core.std.MaskedMerge(deband_a, aa, line)
deband_c = core.f3kdb.Deband(aa, range=12, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_c, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')

grain_a = kgf.adaptive_grain(deband, 0.55, static=False, luma_scaling=6)
grain = fvf.rfs(deband, grain_a, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')

final = fvf.Depth(grain, 10)
final.set_output()