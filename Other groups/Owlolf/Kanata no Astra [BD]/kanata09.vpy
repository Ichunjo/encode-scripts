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

src = lvf.src(r"BDROM Vol 3/BDMV/STREAM/00012.m2ts")

# Audio cuts
asrc = r'BDROM Vol 3/BDMV/STREAM/00012.mka'
ac.eztrim(src,[(0,34047)], asrc,"kanata09.mka")

# Video cuts
src = src[:34047]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata09_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
b_ep, c_ep = 0, 1/2

space =     '[1278 1469] [5644 5732] [8446 8601] [11840 11989] [14785 14946] [15586 15752] [26418 26459] [29223 29361] [30030 30182] [31282 31383]'
flashback = '[0 1136] [10104 10183] [10529 10672] [13049 13294] [18119 18353] [18669 18912] [19036 19163] [21480 21645] [24493 24726]'

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Descale
desc_y_b = dsc.Descale(src32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)

# Denoise luma
lb_m = lvf.wipe_row(src16, None, 1920, 104, 0, 0, 1920, 103, 0, 977, show_mask=True)

den_y_b = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)
den_y_c = core.std.MaskedMerge(fvf.Depth(desc_y_b, 16), den_y_b, vrdf.Bicubic(lb_m, w_ep, h_ep))

den_y = fvf.rfs(den_y_b, den_y_c, mappings=flashback)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(src32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_b = vrdf.Bicubic(nnedi3_rpow2CL(den_y, 2, nns=4), 1920, 1080, 'catrom')

# Anti-aliasing
line_m = kgf.retinex_edgemask(scaled_y_b)

aa_y_a = lvf.upscaled_sraa(scaled_y_b, 1.5, 20, sharp_downscale=False)
aa_y_b = lvf.upscaled_sraa(scaled_y_b, 1.25, sharp_downscale=False)
aa_y = fvf.rfs(aa_y_a, aa_y_b, mappings='[28826 28894]')
aa_y = core.std.MaskedMerge(scaled_y_b, aa_y, line_m)
rep_y = core.edgefixer.ContinuityFixer(aa_y, 4, 0, 4, 0)

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

# Planes merging
merged = core.std.ShufflePlanes([rep_y, aa_uv], [0, 1, 2], vs.YUV)

# Deband
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)
grain_c = core.std.MaskedMerge(db, grain_a, lb_m)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=space)
grain = fvf.rfs(grain, grain_c, mappings=flashback)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(src16, 927, 'bicubic', 0, 1/2, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m, 0), mappings='[0 1087]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m[1367], 0), mappings='[1296 1367]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m[1469], 0), mappings='[1398 1469]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m, 0), mappings='[30030 31365]')

credit = fvf.rfs(credit, src16, mappings='[34015 34046]')

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()