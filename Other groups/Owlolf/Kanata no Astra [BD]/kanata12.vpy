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

src = lvf.src(r"BDROM Vol 4/BDMV/STREAM/00012.m2ts")

# Audio cuts
asrc = r'BDROM Vol 4/BDMV/STREAM/00012.mka'
ac.eztrim(src,[(0,68812)], asrc,"kanata12.mka")

# Video cuts
src = src[:68812]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata12_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
b_ep, c_ep = 0, 1/2

space =     '[556 627] [8545 8583] [8683 8850] [9259 9392] [22585 22704] [24587 24665] [27630 27740] [28359 28457] [29000 29070] [29516 29611] [30293 30469] [32306 32449] [32570 32641] [32900 32967] [34354 34599] [36020 36271] [36743 36847] [38581 38695] [39641 39850] [41388 41518] [65777 65894] [66084 66209] [66510 68811]'
flashback = '[9393 10141] [11285 11433] [14997 15380] [17835 18179] [18860 19108]'

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
grain_c = core.std.MaskedMerge(db, grain_a, lb_m)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=space)
grain = fvf.rfs(grain, grain_c, mappings=flashback)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(src16, 927, 'bicubic', 0, 1/2, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m[90], 0), mappings='[36 127]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m[4482], 0), mappings='[4387 4482]')

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()