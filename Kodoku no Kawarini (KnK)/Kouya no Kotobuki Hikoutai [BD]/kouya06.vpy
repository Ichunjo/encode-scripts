import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
from nnedi3_rpow2 import nnedi3_rpow2
from vsutil import get_y, get_w

core = vs.core
core.max_cache_size = 64000

src = lvf.src(r"Vol.02/BDMV/STREAM/00004.m2ts")[:34046]
src = fvf.Depth(src, 16)

opstart, opend = 0, 2157
eye_start, eye_end = 17190, 17237
edstart, edend = 31528, 33685
edendJikai = 33721
titreend = 33980
end = src.num_frames-1

fixedges_a = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges_b = core.edgefixer.ContinuityFixer(src, [1,0,0], 0, [1,0,0], 0)
fixedges = fvf.rfs(fixedges_a, fixedges_b, mappings=f'[{opstart} {opend}] [{edstart} {edend}]')
fixedges = fvf.rfs(fixedges, src, mappings=f'[{edend+1} {edendJikai}] [{eye_start} {eye_end}] [{titreend} {end}]')

descaled_a = core.descale.Debicubic(fvf.Depth(get_y(fixedges), 32), get_w(842), 842, 0, 1)
descaled_a = core.resize.Spline36(descaled_a, get_w(844), 844)
descaled_b = core.descale.Debicubic(fvf.Depth(get_y(fixedges), 32), get_w(844), 844, 0, 1/2)
descaled = fvf.rfs(descaled_a, descaled_b, mappings=f'[{opstart} {opend}] [{eye_start} {eye_end}] [{edstart} {edend}]')
descaled = fvf.rfs(descaled, descaled_b, mappings='[3653 3936] [4224 4537] [18919 18966] [19331 19550] [19599 19693] [19826 20310] [21701 21891] [25862 26052] [33722 34045]')

denoise_mask = vrdf.adaptive_mask(descaled, 10)
denoise_a = mvf.BM3D(descaled, sigma=0.35, radius1=1)
denoise_b = mvf.BM3D(descaled, sigma=0.75, radius1=1)
denoise = core.std.MaskedMerge(denoise_b, denoise_a, denoise_mask)
denoise = fvf.Depth(denoise, 16)

denoise_cy = mvf.BM3D(descaled, 3.5, radius1=1)
denoise_cc = mvf.BM3D(src, [0, 2], radius1=1)
denoise = fvf.rfs(denoise, denoise_cy, mappings='[2229 2299]')
denoise_c = fvf.rfs(src, denoise_cc, mappings='[2229 2299]')

deband_mask = kgf.retinex_edgemask(denoise)
db_y_a, db_c_a = vrdf.F3kdbSep(denoise, denoise_c, 16, 36, 24, 0, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(denoise, denoise_c, 22, 64, 48, 32, 0, deband_mask)
db_y_c, db_c_c = vrdf.F3kdbSep(denoise, denoise_c, 16, 48, 36, 0, 0)

db_y = fvf.rfs(db_y_a, db_y_b, mappings='[2229 2299]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings='[2229 2299]')
db_y = fvf.rfs(db_y, db_y_c, mappings='[2752 2854] [3587 3652] [3937 3965] [4098 4223]')
db_c = fvf.rfs(db_c, db_c_c, mappings='[2752 2854] [3587 3652] [3937 3965] [4098 4223]')

aa = taa.TAAmbk(db_y, aatype='Eedi3', alpha=0.2, beta=0.25, gamma=1000, nrad=2, mdis=15, mtype=1, mpand=(2,2), sharp=60)

scaled = nnedi3_rpow2(aa, nns=4).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=0)

merged = core.std.ShufflePlanes([scaled, db_c], [0, 1, 2], vs.YUV)

c_mask = vrdf.DRM(fixedges, 844, 'bicubic', b=0, c=1/2, mthr=150, mode='ellipse', sw=4, sh=4).std.Inflate()
credit = fvf.rfs(merged, core.std.MaskedMerge(merged, fixedges, c_mask), mappings=f'[0 195] [{eye_start} {eye_end}] [{edstart} {edend}]')

grain = kgf.adaptive_grain(credit, 0.2, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()