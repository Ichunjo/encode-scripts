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

src = lvf.src(r"BDROM Vol 2v2/BDMV/STREAM/00012.m2ts")

# Audio cuts
asrc = r'BDROM Vol 2v2/BDMV/STREAM/00012.mka'
ac.eztrim(src,[(0,34046)], asrc,"kanata06.mka")

# Video cuts
src = src[:34046]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata06_keyframes.log')

# Variables
w_op, h_op = get_w(720), 720
w_ep, h_ep = get_w(927), 927
w_ed, h_ed = get_w(990), 990
b_op, c_op = 1/3, 1/3
b_ep, c_ep = 0, 1/2
b_ed, c_ed = 1/5, 1/2
opstart, opend = 1, 2157
edstart, edend = 31888, 34045

opening =   f'[{opstart} {opend}]'
ending =    f'[{edstart} {edend}]'
space =     '[22208 22377] [24212 24319] [26545 26791] [27986 28093]'
flashback = '[4119 4778] [13336 13596] [21637 21732]'

# Chroma shift
planes = kgf.split(src)
planes[1], planes[2] = [core.resize.Spline16(plane, src_left=0.5) for plane in planes[1:]]
uv_shift = kgf.join(planes)
uv_shift = fvf.rfs(src, uv_shift, mappings=opening)

# Edges fixer
fe = core.edgefixer.ContinuityFixer(uv_shift, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])
fe = fvf.rfs(uv_shift, fe, mappings=opening)

# Dithers
fe16 = fvf.Depth(fe, 16)
fe32 = fvf.Depth(fe, 32)

# Descale
desc_y_a = dsc.Descale(fe32, w_op, h_op, 'bicubic', b_op, c_op, gray=True)
desc_y_b = dsc.Descale(fe32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)
desc_y_c = dsc.Descale(fe32, w_ed, h_ed, 'bicubic', b_ed, c_ed, gray=True)

# Denoise luma
lb_m = lvf.wipe_row(fe16, None, 1920, 104, 0, 0, 1920, 103, 0, 977, show_mask=True)

den_y_a_a = mvf.BM3D(desc_y_a, 1.25, radius1=1, depth=16)
den_y_a_b = mvf.BM3D(desc_y_a, 2, radius1=1, depth=16)
den_y_b = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)
desc_y_b = fvf.Depth(desc_y_b, 16)
den_y_c = core.std.MaskedMerge(desc_y_b, den_y_b, vrdf.Bicubic(lb_m, w_ep, h_ep))

den_y_a = fvf.rfs(den_y_a_a, den_y_a_b, mappings=f'[{opstart+1138} {opstart+1728}]')
den_y = fvf.rfs(den_y_b, den_y_c, mappings=flashback)

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(fe32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_a = vrdf.Bicubic(nnedi3_rpow2CL(den_y_a, 2, nns=4), 1920, 1080, 'robidoux')
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
merged_a = core.std.ShufflePlanes([scaled_y_a, aa_uv], [0, 1, 2], vs.YUV)
merged_b = core.std.ShufflePlanes([rep_y, aa_uv], [0, 1, 2], vs.YUV)

merged = fvf.rfs(merged_b, merged_a, mappings=opening)

# Deband
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(merged, 22, 80, 60, 60, 32, 0, sample_mode=4)
db_c = core.neo_f3kdb.Deband(merged, 18, 48, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

db = fvf.rfs(db, db_b, mappings=f'[{opstart} {opstart+14}]')
db = fvf.rfs(db, db_c, mappings=f'[{opstart+1138} {opstart+1728}]')

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)
grain_c = core.std.MaskedMerge(db, grain_a, lb_m)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=space + '[10146 10583]' + f'[{opstart+1138} {opstart+1728}]')
grain = fvf.rfs(grain, grain_c, mappings=flashback)
grain = fvf.rfs(grain, grain_d, f'[{opstart+1019} {opstart+1081}]')
grain = fvf.rfs(grain, db, mappings=ending)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(fe16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(fe16, 927, 'bicubic', b_ep, c_ep, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m, 0), mappings='[2245 2325]')

# Ending credits
credit_m = vrdf.DRM(fe16, 990, 'bicubic', b_ed, c_ed, 50, 6, 6).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings=ending)

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()