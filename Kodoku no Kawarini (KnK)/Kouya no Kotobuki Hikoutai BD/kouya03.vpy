import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import havsfunc as hvf
from nnedi3_rpow2CL import nnedi3_rpow2CL
import vardefunc as vrdf
from vsutil import get_y, get_w
core = vs.core
core.max_cache_size = 64000

src = lvf.src(r"Vol.01/BDMV/STREAM/00004.m2ts")[:34047]
src = fvf.Depth(src, 16)

opstart, opend = 1558, 3716
eye_start, eye_end = 13020, 13067
edstart, edend = 31529, 33686
edendJikai = 33722
titreend = 33981
end = src.num_frames-1

fixedges_a = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges_b = core.edgefixer.ContinuityFixer(src, [1,0,0], 0, [1,0,0], 0)
fixedges = fvf.rfs(fixedges_a, fixedges_b, mappings=f'[{opstart} {opend}] [{edstart} {edend}]')
fixedges = fvf.rfs(fixedges, src, mappings=f'[{edend+1} {edendJikai}] [{eye_start} {eye_end}] [{titreend} {end}]')

descaled = core.descale.Debicubic(fvf.Depth(get_y(fixedges), 32), get_w(844), 844, 0, 1/2)
descaled = fvf.Depth(descaled, 16)
denoise_mask = vrdf.adaptive_mask(descaled, 30)
stabilize = hvf.GSMC(descaled, Lmask=denoise_mask, rep=24)

aa = lvf.nneedi3_clamp(stabilize, ret_mask=True)

scaled = nnedi3_rpow2CL(aa, device=0).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=0)

merged = core.std.ShufflePlanes([scaled, src], [0, 1, 2], vs.YUV)

c_mask = vrdf.DRM(fixedges, 844, 'bicubic', b=0, c=1/2, mthr=150)
c_mask = hvf.mt_expand_multi(c_mask, mode='ellipse', sw=2, sh=2).std.Inflate()
credit = fvf.rfs(merged, core.std.MaskedMerge(merged, fixedges, c_mask), mappings=f'[{opstart} {opend}] [{eye_start} {eye_end}] [{edstart} {edend}]')

deband = core.f3kdb.Deband(credit, range=16, y=36, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)

grain = kgf.adaptive_grain(deband, 0.2, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()