import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w, get_y
core = vs.get_core()
core.max_cache_size = 26000

src = lvf.src(r'asuka11.m2ts')[24:34838]
src = fvf.Depth(src, 16)

opstart, opend = 1296, 3452
titlestart, titleend = 3453, 3572
eyestart, eyeend = 17874, 18089
edstart = 32296

crop = core.std.CropRel(src, top=132, bottom=132)
borders = core.std.AddBorders(crop, top=132, bottom=132)
borders = fvf.rfs(src, borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

scaled = fvf.DescaleAA(borders, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)
scaled_title = fvf.DescaleAA(src, get_w(720), 720, kernel='bicubic', b=0, c=1/2)
scaled_eye = fvf.DescaleAA(src, get_w(900), 900, kernel='bicubic', b=0, c=1/2)
scaled = fvf.rfs(scaled, scaled_title, mappings=f'[{titlestart} {titleend}]')
scaled = fvf.rfs(scaled, scaled_eye, mappings=f'[{eyestart} {eyeend}]')
scaled = fvf.rfs(scaled, borders, mappings='[34655 34745]')#################################################C

denoise_a = core.knlm.KNLMeansCL(scaled, a=2, h=0.55, d=3, channels='Y', device_type='gpu')
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise_a = core.std.MaskedMerge(denoise_a, scaled, denoise_mask)
denoise_b = core.knlm.KNLMeansCL(scaled, a=2, h=0.75, d=3, channels='Y', device_type='gpu')
denoise_b = core.knlm.KNLMeansCL(denoise_b, a=2, h=0.45, d=3, channels='UV', device_type='gpu')
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}]')

line = kgf.retinex_edgemask(denoise)

aa_a = lvf.transpose_aa(denoise, eedi3=True)
aa_a = core.std.MaskedMerge(denoise, aa_a, line)
aa_b = lvf.nneedi3_clamp(denoise)
aa_c = lvf.upscaled_sraa(get_y(denoise))
aa_c = core.std.ShufflePlanes([aa_c, denoise], [0, 1, 2], vs.YUV)
mask_aa = lvf.src(r'F:\VPS\ENCO EN COURS\Asuka\mask_ed.png').resize.Point(format=vs.GRAY16, matrix_s="709", chromaloc_s="top_left")
aa_ed = core.std.MaskedMerge(aa_c, denoise, mask_aa)
aa = fvf.rfs(aa_a, aa_b, mappings=f'[{opstart} {opend}]')
aa = fvf.rfs(aa, aa_ed, mappings=f'[{edstart+66} {edstart+125}]')
aa = fvf.rfs(aa, scaled, mappings=f'[{titlestart} {titleend}]')

deband_a = core.f3kdb.Deband(aa, range=17, y=34, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband_a = core.std.MaskedMerge(deband_a, aa, line)
deband_b = core.f3kdb.Deband(aa, range=17, y=48, cb=36, cr=36, grainy=32, grainc=0, output_depth=16, dynamic_grain=True)
deband_c = core.f3kdb.Deband(aa, range=12, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16)
deband_d = core.f3kdb.Deband(aa, range=22, y=64, cb=64, cr=64, grainy=32, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=f'[{opstart+1936} {opstart+1971}]')
deband = fvf.rfs(deband, deband_c, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}] [17433 17510]')
deband = fvf.rfs(deband, deband_d, mappings='[19760 20142] [24131 24190]')

grain_a = kgf.adaptive_grain(deband, 0.55, static=False, luma_scaling=6)
grain = fvf.rfs(deband, grain_a, mappings=f'[{edstart} {edstart+240}] [{edstart+1147} {edstart+1615}] [17433 17510]')

mask_bottom = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=946).std.Inflate()
mask_top = kgf.squaremask(src, width=1920, height=134, offset_x=0, offset_y=0).std.Inflate()
rep = core.std.Expr([mask_bottom, mask_top], 'x y +')
rep_borders = core.std.MaskedMerge(grain, borders, rep)
rep_borders = fvf.rfs(grain, rep_borders, mappings=f'[{opstart+369} {opstart+931}] [{opstart+964} {opstart+1180}]')

final = fvf.Depth(rep_borders, 10)

final.set_output()