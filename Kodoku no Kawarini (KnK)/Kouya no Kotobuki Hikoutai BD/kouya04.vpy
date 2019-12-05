import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
import vsTAAmbk as taa
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_y, get_w
core = vs.core
core.max_cache_size = 64000

def HardMask(src):
    linemask = core.std.Sobel(src).std.Expr("x 2 *").std.Maximum()
    linemask = core.std.CropAbs(linemask, x=858, y=480, width=250, height=158)
    linemask = core.std.AddBorders(linemask, left=858, right=392, top=480, bottom=206)
    return linemask

src = lvf.src(r"Vol.02/BDMV/STREAM/00002.m2ts")[48:34094]
src = fvf.Depth(src, 16)

opstart, opend = 0, 2157
eye_start, eye_end = 13331, 13378
edstart, edend = 31528, 33685
edendJikai = 33721
titreend = 33980
end = src.num_frames-1

fixedges_a = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges_b = core.edgefixer.ContinuityFixer(src, [1,0,0], 0, [1,0,0], 0)
fixedges = fvf.rfs(fixedges_a, fixedges_b, mappings=f'[{opstart} {opend}] [{edstart} {edend}]')
fixedges = fvf.rfs(fixedges, src, mappings=f'[{edend+1} {edendJikai}] [{eye_start} {eye_end}] [{titreend} {end}]')

descaled = core.descale.Debicubic(fvf.Depth(get_y(fixedges), 32), get_w(844), 844, 0, 1/2)
descaled = fvf.Depth(descaled, 16)
stabilize = hvf.GSMC(descaled, rep=24)

db_y_a, db_c_a = vrdf.F3kdbSep(stabilize, src, 16, 36, 24, 0, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(stabilize, src, 16, 48, 36, 0, 0)
db_y = fvf.rfs(db_y_a, db_y_b, mappings='[22265 22392]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings='[22265 22392]')

aa_a = taa.TAAmbk(db_y, aatype='Eedi3', alpha=0.2, beta=0.25, gamma=1000, nrad=2, mdis=15, mtype=1, mpand=(2,2), sharp=60)
aa_b = taa.TAAmbk(aa_a, aatype='Eedi3SangNom', mclip=HardMask(db_y))
aa = fvf.rfs(aa_a, aa_b, mappings='[23530 23548]')

scaled = nnedi3_rpow2CL(aa, nns=4, device=0).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=1/2)

merged = core.std.ShufflePlanes([scaled, db_c], [0, 1, 2], vs.YUV)

c_mask = vrdf.DRM(fixedges, 844, 'bicubic', b=0, c=1/2, mthr=150, mode='ellipse', sw=4, sh=4).std.Inflate()
credit = fvf.rfs(merged, core.std.MaskedMerge(merged, fixedges, c_mask), mappings=f'[{opstart} {opend}] [{eye_start} {eye_end}] [{edstart} {edend}]')

grain = kgf.adaptive_grain(credit, 0.2, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()