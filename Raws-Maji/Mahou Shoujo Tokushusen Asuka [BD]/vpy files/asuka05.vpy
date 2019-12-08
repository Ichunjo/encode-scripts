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

src = lvf.src(r'asuka05.m2ts')[24:35197]
src = fvf.Depth(src, 16)

opstart, opend = 744, 3524
titlestart, titleend = 3800, 3919
eyestart, eyeend = 19649, 19864
edstart, edend = 29252, 31409

crop = core.std.CropRel(src, top=132, bottom=132)
borders = core.std.AddBorders(crop, top=132, bottom=132)
borders = fvf.rfs(src, borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

scaled = fvf.DescaleAA(borders, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_title = fvf.DescaleAA(src, get_w(720), 720, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_eye = fvf.DescaleAA(src, get_w(900), 900, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled = fvf.rfs(scaled, scaled_title, mappings=f'[{titlestart} {titleend}]')
scaled = fvf.rfs(scaled, scaled_eye, mappings=f'[{eyestart} {eyeend}]')
scaled = fvf.rfs(scaled, borders, mappings='[35008 35099]')

denoise_a = vrdf.KNLMeansCL(scaled, 0.55, 0, 0)
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise_a = core.std.MaskedMerge(denoise_a, scaled, denoise_mask)
denoise_b = vrdf.KNLMeansCL(scaled, 0.75, 0.45, 0)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')

line = kgf.retinex_edgemask(denoise)

aa_a = lvf.transpose_aa(denoise, eedi3=True)
aa_a = core.std.MaskedMerge(denoise, aa_a, line)
aa_b = lvf.nneedi3_clamp(denoise)
aa_c = core.std.ShufflePlanes([lvf.upscaled_sraa(get_y(denoise)), denoise], [0, 1, 2], vs.YUV)
mask_aa = lvf.src(r'F:\VPS\ENCO EN COURS\Asuka\mask_ed.png').resize.Point(format=vs.GRAY16, matrix_s="709", chromaloc_s="top_left")
aa_ed = core.std.MaskedMerge(aa_c, denoise, mask_aa)
aa = fvf.rfs(aa_a, aa_b, mappings=f'[{opstart} {opend}]')
#aa = fvf.rfs(aa, aa_c, mappings=f'[{opstart+1955} {opstart+1966}]')
aa = fvf.rfs(aa, aa_ed, mappings=f'[{edstart+66} {edstart+125}]')
aa = fvf.rfs(aa, scaled, mappings=f'[{titlestart} {titleend}]')

deband_a = core.f3kdb.Deband(aa, range=17, y=34, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband_a = core.std.MaskedMerge(deband_a, aa, line)
deband_b = core.f3kdb.Deband(aa, range=17, y=48, cb=36, cr=36, grainy=32, grainc=0, output_depth=16, dynamic_grain=True)
deband_c = core.f3kdb.Deband(aa, range=12, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16)
deband_d = core.f3kdb.Deband(aa, range=19, y=40, cb=40, cr=40, grainy=32, grainc=0, output_depth=16)
deband_e = core.f3kdb.Deband(aa, range=20, y=120, cb=120, cr=120, grainy=32, grainc=0, output_depth=16)
deband_f = core.f3kdb.Deband(aa, range=26, y=240, cb=240, cr=240, grainy=48, grainc=0, output_depth=16)
line_mask = kgf.retinex_edgemask(aa).std.Binarize(8000).rgvs.RemoveGrain(3).std.Inflate().std.Inflate().std.Inflate().std.Inflate()
deband_e = core.std.MaskedMerge(deband_e, aa, line_mask)
deband_f = core.std.MaskedMerge(deband_f, aa, line_mask)
deband = fvf.rfs(deband_a, deband_b, mappings=f'[{opstart+1936} {opstart+1971}]')
deband = fvf.rfs(deband, deband_c, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')
deband = fvf.rfs(deband, deband_d, mappings='[19865 20049] [20863 20988] [21213 21384] [21626 21661] [22060 22440] [22663 22967]')
deband = fvf.rfs(deband, deband_e, mappings='[22968 23259] [23535 23653] [24023 24051]')
deband = fvf.rfs(deband, deband_f, mappings='[23260 23534]')

grain_a = kgf.adaptive_grain(deband, 0.55, static=False, luma_scaling=6)
grain_b = kgf.adaptive_grain(deband, 0.35, static=False, luma_scaling=12)
grain_c = kgf.adaptive_grain(deband, 0.55, static=False, luma_scaling=12)
grain = fvf.rfs(deband, grain_a, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')
grain = fvf.rfs(grain, grain_b, mappings='[22968 23259] [23535 23653] [24023 24051]')
grain = fvf.rfs(grain, grain_c, mappings='[23260 23534]')

mask_bottom = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=946).std.Inflate()
mask_top = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=0).std.Inflate()
rep = core.std.Expr([mask_bottom, mask_top], 'x y +')
rep_borders = core.std.MaskedMerge(grain, borders, rep)
rep_borders = fvf.rfs(grain, rep_borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

final = fvf.Depth(rep_borders, 10)

final.set_output()