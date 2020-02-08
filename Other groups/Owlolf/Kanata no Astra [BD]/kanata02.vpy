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

src = 		lvf.src(r"BDROM Vol 1v2/BDMV/STREAM/00011.m2ts")

# Audio cuts
asrc = r'BDROM Vol 1v2/BDMV/STREAM/00011.mka'
ac.eztrim(src,[(0,34048)], 	asrc,"kanata02.mka")

# Video cuts
src = src[:34048]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata02_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
b_ep, c_ep = 0, 1/2

space =     '[0 107] [12770 12925]'
flashback = '[3597 3668] [13640 13957] [16801 17238] [17698 18703] [27931 29136]'

# Dithers
fe16 = fvf.Depth(src, 16)
fe32 = fvf.Depth(src, 32)

# Descale
desc_y = dsc.Descale(fe32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)

# Denoise luma
lb_m = lvf.wipe_row(fe16, None, 1920, 104, 0, 0, 1920, 103, 0, 977, show_mask=True)

den_y_b = mvf.BM3D(desc_y, 2.2, radius1=1, profile1='lc', depth=16)
den_y_c = core.std.MaskedMerge(fvf.Depth(desc_y, 16), den_y_b, vrdf.Bicubic(lb_m, w_ep, h_ep))

den_y = fvf.rfs(den_y_b, den_y_c, mappings=flashback)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(fe32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_b = vrdf.Bicubic(nnedi3_rpow2CL(den_y, 2, nns=4), 1920, 1080, 'catrom')

# Anti-aliasing
line_m = kgf.retinex_edgemask(scaled_y_b)

aa_y = lvf.upscaled_sraa(scaled_y_b, 1.5, 20, sharp_downscale=False)
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

grain = fvf.rfs(grain_a, grain_b, mappings=space)
grain = fvf.rfs(grain, grain_c, mappings=flashback)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(fe16, 1.5, rep=13, sharp_downscale=False)

credit_m1 = vrdf.DRM(fe16, 927, 'bicubic', 0, 1/2, 20, 8, 8).std.Inflate()
credit_m2 = vrdf.DRM(fe16, 927, 'bicubic', 0, 1/2, 80, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m1[815], 0), mappings='[752 858]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m1[920], 0), mappings='[864 1007]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m2, 0), mappings='[0 743] [31133 33021] [33977 34047]')

# Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()