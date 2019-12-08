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

src = lvf.src(r'asuka01.m2ts')[24:34838]
src = fvf.Depth(src, 16)

opstart, opend = 32296, 34453
titlestart, titleend = 6717, 6836
eyestart, eyeend = 16100, 16316

crop = core.std.CropRel(src, top=132, bottom=132)
borders = core.std.AddBorders(crop, top=132, bottom=132)
borders = fvf.rfs(src, borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

scaled = fvf.DescaleAA(borders, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_title = fvf.DescaleAA(src, get_w(720), 720, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_eye = fvf.DescaleAA(src, get_w(900), 900, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled = fvf.rfs(scaled, scaled_title, mappings=f'[{titlestart} {titleend}]')
scaled = fvf.rfs(scaled, scaled_eye, mappings=f'[{eyestart} {eyeend}]')
scaled = fvf.rfs(scaled, borders, mappings='[34644 34717]')

denoise_a = vrdf.KNLMeansCL(scaled, 0.55, 0, 0)
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise = core.std.MaskedMerge(denoise_a, scaled, denoise_mask)

line = kgf.retinex_edgemask(denoise)

aa_a = lvf.transpose_aa(denoise, eedi3=True)
aa_a = core.std.MaskedMerge(denoise, aa_a, line)
aa_b = lvf.nneedi3_clamp(denoise)
aa_c = core.std.ShufflePlanes([lvf.upscaled_sraa(get_y(denoise)), denoise], [0, 1, 2], vs.YUV)
aa = fvf.rfs(aa_a, aa_b, mappings=f'[{opstart} {opend}]]')
aa = fvf.rfs(aa, aa_c, mappings=f'[{opstart+1955} {opstart+1966}]')
aa = fvf.rfs(aa, scaled, mappings=f'[{titlestart} {titleend}]')

deband_a = core.f3kdb.Deband(aa, range=17, y=34, cb=26, cr=26, grainy=16, grainc=0, output_depth=16)
deband_a = core.std.MaskedMerge(deband_a, aa, line)
deband_b = core.f3kdb.Deband(aa, range=17, y=48, cb=36, cr=36, grainy=32, grainc=0, output_depth=16, dynamic_grain=True)
deband_c = core.f3kdb.Deband(aa, range=12, y=64, cb=56, cr=56, grainy=0, grainc=0, output_depth=16, dynamic_grain=True)
deband_c = core.f3kdb.Deband(deband_c, range=18, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16, dynamic_grain=True)
deband_d = core.f3kdb.Deband(aa, range=20, y=96, cb=80, cr=80, grainy=48, grainc=0, output_depth=16, dynamic_grain=True)
deband = fvf.rfs(deband_a, deband_b, mappings=f'[{opstart+1936} {opstart+1948}] [{opstart+1975} {opstart+1988}]')
deband = fvf.rfs(deband, deband_c, mappings=f'[{opstart+1949} {opstart+1954}]')
deband = fvf.rfs(deband, deband_d, mappings=f'[{opstart+1967} {opstart+1974}]')

mask_bottom = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=946).std.Inflate()
mask_top = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=0).std.Inflate()
rep = core.std.Expr([mask_bottom, mask_top], 'x y +')
rep_borders = core.std.MaskedMerge(deband, borders, rep)
rep_borders = fvf.rfs(deband, rep_borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

final = fvf.Depth(rep_borders, 10)
final.set_output()