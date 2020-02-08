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

src = lvf.src(r"BDROM Vol 1v2/BDMV/STREAM/00010.m2ts")

# Audio cuts
asrc = r'BDROM Vol 1v2/BDMV/STREAM/00010.mka'
ac.eztrim(src,[(24,-26)], 	asrc,"kanata01_a.mka")
ac.eztrim(src,[(0,2)], 		asrc,"kanata01_b.mka")

# Video cuts
src = 		src[24:-26]
src = 		src+(src[-1])*2

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata01_keyframes.log')

# Variables
w_op, h_op = get_w(720), 720
w_ep, h_ep = get_w(927), 927
b_op, c_op = 1/3, 1/3
b_ep, c_ed = 0, 1/2
opstart, opend = 66413, 68571

opening = f'[{opstart} {opend}]'
space = '[0 365] [450 820] [906 1193] [1278 1720] [1805 2591] [19924 20091] [20224 20319] [25529 25636] [30861 30907] [32423 35288] [36138 36341] [37083 37346] [39025 39132] [40011 40116] [40177 40916] [40962 41077] [41168 41365] [41390 41413] [42170 42214] [42311 42574] [42659 42749] [42989 43120] [43253 43435] [43766 43885] [45221 45364] [45546 45617] [45792 45851] [45870 45929] [46154 46419] [46938 47242] [56794 56899] [57160 57288] [63462 63557] [65961 66412]'
flashback = '[30980 32422] [38017 38121] [41366 41389] [41414 41622] [41815 41874] [45437 45545] [55603 55992] [56195 56448]'

# Edges fixer
fe = core.edgefixer.ContinuityFixer(src, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])
fe = fvf.rfs(src, fe, mappings=opening)

# Dithers
fe16 = fvf.Depth(fe, 16)
fe32 = fvf.Depth(fe, 32)

# Descale
desc_y_a = dsc.Descale(fe32, w_op, h_op, 'bicubic', b_op, c_op, gray=True)
desc_y_b = dsc.Descale(fe32, w_ep, h_ep, 'bicubic', b_ep, c_ed, gray=True)

# Denoise luma
lb_m = lvf.wipe_row(fe16, None, 1920, 104, 0, 0, 1920, 103, 0, 977, show_mask=True)

den_y_a = mvf.BM3D(desc_y_a, 1.25, radius1=1, depth=16)
den_y_b = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)
desc_y_b = fvf.Depth(desc_y_b, 16)
den_y_c = core.std.MaskedMerge(desc_y_b, den_y_b, vrdf.Bicubic(lb_m, w_ep, h_ep))

den_y = fvf.rfs(den_y_b, den_y_c, mappings=flashback)
den_y = fvf.rfs(den_y, desc_y_b, mappings='[42845 42904]')

# Denoise chroma
den_uv = core.knlm.KNLMeansCL(fe32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

# Rescale
scaled_y_a = vrdf.Bicubic(nnedi3_rpow2CL(den_y_a, 2, nns=4), 1920, 1080, 'robidoux')
scaled_y_b = vrdf.Bicubic(nnedi3_rpow2CL(den_y, 2, nns=4), 1920, 1080, 'catrom')

# Anti-aliasing
line_m = kgf.retinex_edgemask(scaled_y_b)

aa_y = lvf.upscaled_sraa(scaled_y_b, 1.5, 20, sharp_downscale=False)
aa_y = core.std.MaskedMerge(scaled_y_b, aa_y, line_m)
rep_y = core.edgefixer.ContinuityFixer(aa_y, 4, 0, 4, 0)

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

# Planes merging
merged_a = core.std.ShufflePlanes([scaled_y_a, aa_uv], [0, 1, 2], vs.YUV)
merged_b = core.std.ShufflePlanes([rep_y, aa_uv], [0, 1, 2], vs.YUV)

merged = fvf.rfs(merged_b, merged_a, mappings=opening)

# Deband
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(merged, 22, 80, 60, 60, 32, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

db = fvf.rfs(db, db_b, mappings=f'[{opstart} {opstart+14}]')

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)
grain_c = core.std.MaskedMerge(db, grain_a, lb_m)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=space + '[41623 41814]')
grain = fvf.rfs(grain, grain_c, mappings=flashback)
grain = fvf.rfs(grain, grain_d, f'[{opstart+1019} {opstart+1081}]')

# Random 1080p things
aa_cred = lvf.upscaled_sraa(fe16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(fe16, 927, 'bicubic', 0, 1/2, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m[3011], 0), mappings='[2955 3062]')

# Credit OP
credit_m = vrdf.DRM(fe16, 720, 'Spline36', None, None, 100, 4, 4).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings=opening)

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()