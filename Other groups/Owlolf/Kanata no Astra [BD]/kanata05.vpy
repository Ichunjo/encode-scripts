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

src = lvf.src(r"BDROM Vol 2v2/BDMV/STREAM/00011.m2ts")

# Audio cuts
asrc = r'BDROM Vol 2v2/BDMV/STREAM/00011.mka'
ac.eztrim(src,[(0,34046)], asrc,"kanata05.mka")

# Video cuts
src = src[:34046]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata05_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
w_ed, h_ed = get_w(990), 990
b_ep, c_ep = 0, 1/2
b_ed, c_ed = 1/5, 1/2
edstart, edend = 32285, 34045

ending =    f'[{edstart} {edend}]'
space =     '[4120 4347] [13651 13838] [14385 14527]'

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Descale
desc_y_b = dsc.Descale(src32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)
desc_y_c = dsc.Descale(src32, w_ed, h_ed, 'bicubic', b_ed, c_ed, gray=True)

# Denoise luma
den_y = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(src32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_b = vrdf.Bicubic(nnedi3_rpow2CL(den_y, 2, nns=4), 1920, 1080, 'catrom')
scaled_y_c = vrdf.Bicubic(nnedi3_rpow2CL(fvf.Depth(desc_y_c, 16), 2, nns=4), 1920, 1080, 'catrom')

scaled_y_bc = fvf.rfs(scaled_y_b, scaled_y_c, mappings=ending)

# Anti-aliasing
line_m = kgf.retinex_edgemask(scaled_y_bc)

aa_y = lvf.upscaled_sraa(scaled_y_bc, 1.5, 20, sharp_downscale=False)
aa_y = core.std.MaskedMerge(scaled_y_bc, aa_y, line_m)
rep_y = core.edgefixer.ContinuityFixer(aa_y, 4, 0, 4, 0)

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

# Planes merging
merged = core.std.ShufflePlanes([rep_y, aa_uv], [0, 1, 2], vs.YUV)

# Extra AA
aa_a = taa.TAAmbk(merged, aatype='Eedi3SangNom', cycle=1, down8=False, mtype=2)
merged = fvf.rfs(merged, aa_a, mappings='[892 966]')

# Deband
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)

grain = fvf.rfs(grain_a, grain_b, mappings=space)
grain = fvf.rfs(grain, db, mappings=ending)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(src16, 927, 'bicubic', b_ep, c_ep, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m, 0), mappings='[185 656] [1226 2778]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m, 0), mappings='[14209 14309]')

# Ending credits
credit_m = vrdf.DRM(src16, 990, 'bicubic', b_ed, c_ed, 50, 6, 6).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m, 0), mappings=ending)

# Borders 
crop = core.std.Crop(credit, 160, 160, 0, 0)
borders = core.std.AddBorders(crop, 160, 160, 0, 0)
borders = fvf.rfs(credit, borders, mappings='[5035 5691]')

#Final Dither
final = fvf.Depth(borders, 10)

# Output
final.set_output()