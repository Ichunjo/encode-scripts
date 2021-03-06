import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import havsfunc as hvf
import descale as dsc
import debandshit as dbs
import mvsfunc as mvf
import acsuite
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 32384

ac = acsuite.AC()

src = 		lvf.src(r'[BDMV][191127][Bokutachi wa Benkyou ga Dekinai][Vol.6]/BDMV/STREAM/00003.m2ts')
src_ncop = 	lvf.src(r'[BDMV][190626][Bokutachi wa Benkyou ga Dekinai][Vol.1]/BDMV/STREAM/00004.m2ts')[24:-24]
src_nced = 	lvf.src(r'[BDMV][190626][Bokutachi wa Benkyou ga Dekinai][Vol.1]/BDMV/STREAM/00005.m2ts')[24:-24]
asrc = r'[BDMV][191127][Bokutachi wa Benkyou ga Dekinai][Vol.6]/BDMV/STREAM/00003.mka'
 
ac.eztrim(src,[(0,34046)], asrc,"bokuben13.mka")
src = src[:34046]
opstart, opend = 3021, 5178
edstart, edend = 31313, 33469
opening_a = f'[{opstart+400} {opstart+420}]'
opening_b = f'[{opstart+1613} {opstart+1639}]'

src = fvf.Depth(src, 16)

# Edges fixe
fe = core.edgefixer.ContinuityFixer(src, 1, 1, 1, 1)

# Chroma shift
shift = core.fmtc.resample(fe, sx=[0, 0.5], kernel='bicubic')

# Descale
planes = kgf.split(shift)
desc_y = dsc.Descale(planes[0], kgf.get_w(864), 864, 'bicubic', 0, 1/2)

# First AA 
aa_y = lvf.nneedi3_clamp(desc_y, ret_mask=True, alpha=0.4, beta=0.4)

# Variables
upw4 = round(kgf.get_w(864) * 0.375) * 12
uph4 = round(864 * 0.375) * 12
eeargs = dict(field=1, dh=True, alpha=0.6, beta=0.2, gamma=40, nrad=3, mdis=30)
znargs = dict(field=1, dh=True, nsize=2, nns=4)

# Upscale + AA
aa_y = core.eedi3m.EEDI3(aa_y, **eeargs, sclip=core.znedi3.nnedi3(aa_y, **znargs))
aa_y = core.resize.Bicubic(aa_y, desc_y.width, uph4, src_top=-0.5, filter_param_a=0, filter_param_b=1/2).std.Transpose()
aa_y = core.eedi3m.EEDI3(aa_y, **eeargs, sclip=core.znedi3.nnedi3(aa_y, **znargs))
aa_y = core.resize.Bicubic(aa_y, uph4, upw4, src_top=-0.5, filter_param_a=0, filter_param_b=1/2)
aa_y = core.sangnom.SangNom(aa_y, aa=18).std.Transpose()
aa_y = core.sangnom.SangNom(aa_y, aa=18)

scaled_y = core.resize.Spline36(aa_y, 1920, 1080)
contra_y = hvf.LSFmod(scaled_y, strength=30, defaults='slow', source=planes[0])

line_m = TAAmbk(shift, mtype=1, mpand=(1,1), showmask=1)
line_m = line_m.std.Binarize(9000).std.Deflate().std.Inflate()
merged_y = core.std.MaskedMerge(planes[0], contra_y, line_m, 0)

aa_uv = TAAmbk(shift, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3')
scaled = core.std.ShufflePlanes([merged_y, aa_uv], [0, 1, 2], vs.YUV)

aa = TAAmbk(shift, aatype='Eedi3', cycle=2)
scaled = fvf.rfs(scaled, aa, mappings=opening_b)

# Denoise
den_a = CoolDegrain(scaled, tr=1, thsad=60, thsadc=60, blksize=8, overlap=4)
den_b = mvf.BM3D(den_a, [7, 0], radius1=1)

diff_y = core.std.MakeDiff(den_a, den_b, [0])

# Deband masks
adapt_m = vrdf.adaptive_mask(den_b, 25)
line_m = kgf.retinex_edgemask(den_b, 0.1).std.Binarize(9800)
line_m = line_m.rgvs.RemoveGrain(3).std.Inflate()

# Deband phase 1
db_a = core.neo_f3kdb.Deband(den_b, 17, 36, 48, 48, 0, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(den_b, 17, 48, 48, 48, 0, 0, sample_mode=4)
db_b = core.std.MaskedMerge(db_a, db_b, adapt_m)

db = lvf.limit_dark(db_a, db_b, threshold=0.2)
db = core.std.MaskedMerge(db, db_a, line_m, 0)
db = core.std.MergeDiff(db, diff_y, [0])

# Deband phase 2
db_c = dbs.f3kbilateral(db, 17, 64, 64)
db = fvf.rfs(db, db_c, mappings=opening_a)

# Grain by neo_f3kdb
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')

# Then grain by grain.AddGrain
grain = kgf.adaptive_grain(db, 0.3, luma_scaling=8)

# 1080p things
ref = shift
credit_op_m = vrdf.DCM(grain, src[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 2, 2)
credit_ed_m = vrdf.DCM(grain, src[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 2, 2)
credit_m = core.std.Expr([credit_op_m, credit_ed_m], 'x y +')
credit = core.std.MaskedMerge(grain, ref, credit_m)

credit_m = vrdf.DRM(planes[0], 864, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m), mappings='[5192 5309]')

credit = fvf.rfs(credit, shift, mappings='[33926 34045]')

final = fvf.Depth(credit, 10)

final.set_output()