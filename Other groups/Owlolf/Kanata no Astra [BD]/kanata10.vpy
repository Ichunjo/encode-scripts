import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import mvsfunc as mvf
import vsTAAmbk as taa
import vardefunc as vrdf
import descale as dsc
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_w

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r"BDROM Vol 4/BDMV/STREAM/00010.m2ts")

# Audio cuts
asrc = r'BDROM Vol 4/BDMV/STREAM/00010.mka'
ac.eztrim(src,[(24,34070)], asrc,"kanata10.mka")

# Video cuts
src = src[24:34070]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata10_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
b_ep, c_ep = 0, 1/2

space =     '[398 542] [9277 9743] [10934 11026] [11124 11192] [11295 11524] [11576 11717] [13218 13524] [15610 15811] [31509 31675]'

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Descale
desc_y_b = dsc.Descale(src32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)

# Denoise luma
den_y = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(src32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_b = vrdf.Bicubic(nnedi3_rpow2CL(den_y, 2, nns=4), 1920, 1080, 'catrom')

# Anti-aliasing
line_m = kgf.retinex_edgemask(scaled_y_b)

aa_y = lvf.upscaled_sraa(scaled_y_b, 1.5, 20, sharp_downscale=False)
aa_y = core.std.MaskedMerge(scaled_y_b, aa_y, line_m)

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

# Planes merging
merged = core.std.ShufflePlanes([aa_y, aa_uv], [0, 1, 2], vs.YUV)

# Deband
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=space)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(src16, 927, 'bicubic', 0, 1/2, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m, 0), mappings='[546 1734]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m[2917], 0), mappings='[2862 2941]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m[2996], 0), mappings='[2951 3021]')
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m, 0), mappings='[32985 34045]')

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()