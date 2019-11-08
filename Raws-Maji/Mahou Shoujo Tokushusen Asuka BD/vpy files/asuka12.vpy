import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w, get_y
core = vs.get_core()
core.max_cache_size = 26000

src = lvf.src(r'asuka12.m2ts')[24:34836]
src = fvf.Depth(src, 16)
src = src+src[-1]+src[-1]

titlestart, titleend = 1932, 2051
eyestart, eyeend = 17060, 17275
edstart = 32296

scaled = fvf.DescaleAA(src, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_title = fvf.DescaleAA(src, get_w(720), 720, kernel='bicubic', b=0, c=1/2)
scaled_eye = fvf.DescaleAA(src, get_w(900), 900, kernel='bicubic', b=0, c=1/2)
scaled = fvf.rfs(scaled, scaled_title, mappings=f'[{titlestart} {titleend}]')
scaled = fvf.rfs(scaled, scaled_eye, mappings=f'[{eyestart} {eyeend}]')

denoise_a = core.knlm.KNLMeansCL(scaled, a=2, h=0.55, d=3, channels='Y', device_type='gpu')
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise = core.std.MaskedMerge(denoise_a, scaled, denoise_mask)

line = kgf.retinex_edgemask(denoise)

aa_a = lvf.transpose_aa(denoise, eedi3=True)
aa_a = core.std.MaskedMerge(denoise, aa_a, line)
aa = fvf.rfs(aa_a, scaled, mappings=f'[{titlestart} {titleend}]')

deband_a = core.f3kdb.Deband(aa, range=17, y=34, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_a, aa, line)

final = fvf.Depth(deband, 10)
final.set_output()