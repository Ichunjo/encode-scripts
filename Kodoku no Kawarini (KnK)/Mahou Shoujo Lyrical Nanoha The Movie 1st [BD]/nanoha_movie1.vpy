import vapoursynth as vs
import havsfunc as hvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import acsuite
from cooldegrain import CoolDegrain
from nnedi3_rpow2CL import nnedi3_rpow2CL

core = vs.core
core.max_cache_size = 8192
ac = acsuite.AC()

src = lvf.src(r'魔法少女リリカルなのは The Movie First/本篇/BDROM/BDMV/STREAM/00000.m2ts')
asrc_a = r'魔法少女リリカルなのは The Movie First/本篇/BDROM/BDMV/STREAM/nanoha_movie1_2.0.wav'
asrc_b = r'魔法少女リリカルなのは The Movie First/本篇/BDROM/BDMV/STREAM/nanoha_movie1_5.1.wav'
ac.eztrim(src,[(24,-48)], asrc_a,"nanoha_movie1_2.0.mka")
ac.eztrim(src,[(24,-48)], asrc_b,"nanoha_movie1_5.1.mka")

src = src[24:-48]
src = fvf.Depth(src, 16)

fe = core.edgefixer.ContinuityFixer(src, [2,1,1], [2,1,1], [2,1,1], [2,1,1])

# # # # # # # # # # 
desc_str = 0.85
y = fvf.Depth(lvf.plane(fe, 0), 32)
desc_y = core.descale.Debicubic(y, 1280, 720, 1/5, 1/2, cache_size=5)
down_y = core.resize.Bicubic(y, 1280, 720)
desc_y = core.std.Expr([desc_y, down_y], f'x {desc_str} * y 1 {desc_str} - * +')


# # # # # # # # # # 
aa_y = nnedi3_rpow2CL(desc_y, 2, 1280, 720, kernel='spline64', nsize=1, nns=4, qual=2)
aa_y = fvf.Depth(aa_y, 16)

# # # # # # # # 
dering_y = hvf.EdgeCleaner(aa_y, strength=5, rmode=13, smode=1, hot=True)

# # # # # # # # 
credit_m = vrdf.DRM(y, 720, 'bicubic', 1/5, 1/2, 120)
credit_m = vrdf.RegionMask(credit_m, 8, 8, 8, 8).std.Inflate().resize.Spline36(1280, 720)
credit_y = core.std.MaskedMerge(dering_y, fvf.Depth(down_y, 16), fvf.Depth(credit_m, 16))


# # # # # # # # 
den_uv = CoolDegrain(fe, tr=1, thsad=84, blksize=8, overlap=4, plane=3)
u, v  = vrdf.to444(den_uv, 1280, 720, False)

scaled = lvf.join([credit_y, u, v])

# # # # # # # # 
db = core.neo_f3kdb.Deband(scaled, 15, 36, 36, 36, 12, 0, sample_mode=4)
grain = kgf.adaptive_grain(db, 0.25)

# # # # # # # # # # 
cred = lvf.upscaled_sraa(lvf.plane(src, 0), 2, 6, 720, 16/9, False)
cred = lvf.join([cred, u, v])
cred = core.neo_f3kdb.Deband(cred, 16, 64, 48, 48, 0, 0, sample_mode=4)

credit = fvf.rfs(grain, cred, mappings='[0 215] [181849 187223]')

# # # # # # # # 
final = core.fmtc.bitdepth(credit, bits=10, dmode=3)

final.set_output()