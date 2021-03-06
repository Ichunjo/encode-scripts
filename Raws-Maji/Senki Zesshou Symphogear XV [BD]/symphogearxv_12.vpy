import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
import vardefunc as vrdf
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 8192

src = lvf.src(r'SYMPHOGEAR_XV_VOL6/BDMV/STREAM/00005.m2ts')

src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)

# # # # # # # # 
y = lvf.get_y(den)

znargs = dict(field=1, dh=False, nsize=3, nns=3, qual=2)
eeargs = dict(field=0, dh=True, alpha=0.6, beta=0.2, gamma=40)
aa_y = core.std.Transpose(y)
aa_y = core.eedi3m.EEDI3(aa_y, **eeargs).znedi3.nnedi3(**znargs)
aa_y = core.sangnom.SangNom(aa_y, aa=16).sangnom.SangNom(aa=16)
aa_y = core.resize.Bicubic(aa_y, den.height, den.width, src_top=0.5)

aa_y = core.std.Transpose(aa_y)
aa_y = core.eedi3m.EEDI3(aa_y, **eeargs).znedi3.nnedi3(**znargs)
aa_y = core.resize.Bicubic(aa_y, den.width, den.height, src_top=0.5)

line_m = TAAmbk(den, mtype=1, showmask=True).std.Expr("x 2 *").std.Maximum()
aa_y = core.std.MaskedMerge(y, aa_y, line_m)

aa_uv = TAAmbk(den, aatype=0, aatypeu='Eedi3', aatypev='Eedi3')

aa = core.std.ShufflePlanes([aa_y, aa_uv], [0, 1, 2], vs.YUV)

# # # # # # 
blur = core.std.Convolution(aa, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0,1,2])
sharp_lo = core.std.Expr(clips=[aa, blur], expr=['x y - ' + str(0.75) + ' * x +'])
sharp_hi = core.std.Expr(clips=[aa, blur], expr=['x y - ' + str(1.5) + ' * x +'])

adapt_m = core.adg.Mask(aa.std.PlaneStats(), 12)
adapt_m = lvf.iterate(adapt_m, core.std.Minimum, 4)
adapt_m = core.bilateral.Gaussian(adapt_m, 10)

sharp = core.std.MaskedMerge(sharp_lo, sharp_hi, adapt_m)

line_m = core.std.Minimum(line_m)
line_m = lvf.iterate(line_m, core.std.Deflate, 2)

sharp = core.std.MaskedMerge(aa, sharp, line_m)
sharp = core.rgvs.Repair(sharp, aa, 13)

# # # # # # # 
line_m = mdf.retinex_edgemask_mod(sharp).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(sharp, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, sharp, line_m)

grain = kgf.adaptive_grain(db, 0.3, luma_scaling=6)

# # # # # # 
ref = den
out = grain

credit_m = vrdf.DRM(src, 873, 'Spline36', 0, 0, 120).std.Inflate()
credit = fvf.rfs(out, core.std.MaskedMerge(out, ref, credit_m), mappings='[252 2084] [2159 2240] [11640 11805]')

credit = fvf.rfs(credit, src, mappings='[31646 34043]')

# # # # # 
final = core.fmtc.bitdepth(credit, bits=10, dmode=3)

final = final+final[-1]*2

final.set_output()