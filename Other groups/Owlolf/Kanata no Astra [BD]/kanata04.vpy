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
from adjust import Tweak

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r"BDROM Vol 2v2/BDMV/STREAM/00010.m2ts")

# Audio cuts
asrc = r'BDROM Vol 2v2/BDMV/STREAM/00010.mka'
ac.eztrim(src,[(24,34070)], asrc,"kanata04.mka")

# Video cuts
src = src[24:34070]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata04_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
b_ep, c_ep = 0, 1/2

flashback = '[16930 17883] [20281 20394]'

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
grain_c = core.std.MaskedMerge(db, grain_a, lb_m)

grain = fvf.rfs(grain_a, grain_c, mappings=flashback)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m1 = vrdf.DRM(src16, 927, 'bicubic', 0, 1/2, 20, 8, 8).std.Inflate()
credit_m2 = vrdf.DRM(src16, 927, 'bicubic', 0, 1/2, 80, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m2, 0), mappings='[174 461] [1313 1963] [3357 3600] [31730 31813] [32051 33439] [33998 34045]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m1[3692], 0), mappings='[3609 3692]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m1[3956], 0), mappings='[3837 3956]')

# Why??
dimming = Tweak(credit, sat=1.2, cont=1.2)
dimming = fvf.rfs(credit, dimming, mappings='[27992 28151]')

# Final Dither
final = fvf.Depth(dimming, 10)

# Output
final.set_output()