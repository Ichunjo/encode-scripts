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

def _rep_borders(clip, top, bottom):
	clip = core.std.Crop(clip, 0, 0, top, bottom)
	clip = core.std.AddBorders(clip, 0, 0, top, bottom)
	return clip

src = lvf.src(r"BDROM Vol 3/BDMV/STREAM/00010.m2ts")

# Audio cuts
asrc = r'BDROM Vol 3/BDMV/STREAM/00010.mka'
ac.eztrim(src,[(24,34070)], asrc,"kanata07.mka")

# Video cuts
src = src[24:34069]
src = src+src[-1]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'kanata07_keyframes.log')

# Variables
w_ep, h_ep = get_w(927), 927
w_ed, h_ed = get_w(990), 990
b_ep, c_ep = 0, 1/2
b_ed, c_ed = 1/5, 1/2
edstart, edend = 31885, 34045

ending =    f'[{edstart} {edend}]'
space =     '[11118 11221] [11534 11674]'
flashback = '[3984 8650]'

# Fuckings moving border
# I'm sure there is a better method
borders_a = _rep_borders(src, 104, 102)
borders_b = _rep_borders(src, 108, 102)
borders_c = _rep_borders(src, 112, 102)
borders_d = _rep_borders(src, 116, 102)
borders_e = _rep_borders(src, 120, 102)
borders_f = _rep_borders(src, 124, 102)
borders_g = _rep_borders(src, 128, 102)
borders_h = _rep_borders(src, 132, 102)
borders_i = _rep_borders(src, 136, 102)
borders_j = _rep_borders(src, 140, 102)
borders_k = _rep_borders(src, 144, 102)
borders_l = _rep_borders(src, 148, 102)
borders_m = _rep_borders(src, 152, 102)
borders_n = _rep_borders(src, 156, 102)
borders_o = _rep_borders(src, 160, 102)
borders_p = _rep_borders(src, 164, 102)
borders_q = _rep_borders(src, 168, 102)
borders_r = _rep_borders(src, 172, 102)
borders_s = _rep_borders(src, 176, 102)
borders_t = _rep_borders(src, 180, 102)
borders_u = _rep_borders(src, 184, 102)
borders_v = _rep_borders(src, 188, 102)
borders_w = _rep_borders(src, 192, 102)
borders_x = _rep_borders(src, 196, 102)
borders_y = _rep_borders(src, 200, 102)
borders_z = _rep_borders(src, 204, 102)
borders_aa = _rep_borders(src, 208, 102)
borders_ab = _rep_borders(src, 212, 102)
src = fvf.rfs(src, borders_a, mappings='[14506 14550]')
src = fvf.rfs(src, borders_b, mappings='14551')
src = fvf.rfs(src, borders_c, mappings='14552')
src = fvf.rfs(src, borders_d, mappings='14553')
src = fvf.rfs(src, borders_e, mappings='14554')
src = fvf.rfs(src, borders_f, mappings='14555')
src = fvf.rfs(src, borders_g, mappings='14556')
src = fvf.rfs(src, borders_h, mappings='14557')
src = fvf.rfs(src, borders_i, mappings='14558')
src = fvf.rfs(src, borders_j, mappings='14559')
src = fvf.rfs(src, borders_k, mappings='14560')
src = fvf.rfs(src, borders_l, mappings='14561')
src = fvf.rfs(src, borders_m, mappings='14562')
src = fvf.rfs(src, borders_n, mappings='14563')
src = fvf.rfs(src, borders_o, mappings='14564')
src = fvf.rfs(src, borders_p, mappings='14565')
src = fvf.rfs(src, borders_q, mappings='14566')
src = fvf.rfs(src, borders_r, mappings='14567')
src = fvf.rfs(src, borders_s, mappings='14568')
src = fvf.rfs(src, borders_t, mappings='14569')
src = fvf.rfs(src, borders_u, mappings='14570')
src = fvf.rfs(src, borders_v, mappings='14571')
src = fvf.rfs(src, borders_w, mappings='14572')
src = fvf.rfs(src, borders_x, mappings='14573')
src = fvf.rfs(src, borders_y, mappings='14574')
src = fvf.rfs(src, borders_z, mappings='14575')
src = fvf.rfs(src, borders_aa, mappings='14576')
src = fvf.rfs(src, borders_ab, mappings='14577')

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Descale
desc_y_b = dsc.Descale(src32, w_ep, h_ep, 'bicubic', b_ep, c_ep, gray=True)
desc_y_c = dsc.Descale(src32, w_ed, h_ed, 'bicubic', b_ed, c_ed, gray=True)

# Denoise luma
lb_m = lvf.wipe_row(src16, None, 1920, 104, 0, 0, 1920, 103, 0, 977, show_mask=True)

den_y_b = mvf.BM3D(desc_y_b, 2.2, radius1=1, profile1='lc', depth=16)
desc_y_b = fvf.Depth(desc_y_b, 16)
den_y_c = core.std.MaskedMerge(desc_y_b, den_y_b, vrdf.Bicubic(lb_m, w_ep, h_ep))

den_y = fvf.rfs(den_y_b, den_y_c, mappings=flashback)

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

grain = fvf.rfs(grain_a, grain_b, mappings=space)
grain = fvf.rfs(grain, grain_c, mappings=flashback)
grain = fvf.rfs(grain, db, mappings=ending)

# Random 1080p things
aa_cred = lvf.upscaled_sraa(src16, 1.5, rep=13, sharp_downscale=False)

credit_m = vrdf.DRM(src16, 927, 'bicubic', b_ep, c_ep, 50, 6, 6).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m, 0), mappings='[0 639]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa_cred, credit_m, 0), mappings='[640 759]')

# Ending credits
credit_m = vrdf.DRM(src16, 990, 'bicubic', b_ed, c_ed, 50, 6, 6).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m, 0), mappings=ending)

#Final Dither
final = fvf.Depth(credit, 10)

# Output
final.set_output()