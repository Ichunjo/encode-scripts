import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import acsuite
import psutil
from cooldegrain import CoolDegrainSF
from math import ceil
p_handle = psutil.Process()
p_handle.cpu_affinity(list(range(0,23,2)))

core = vs.core
core.max_cache_size = 26000
core.num_threads = 12

ac = acsuite.AC()

src = core.dgdecodenv.DGSource(r"[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc1/BDMV/STREAM/00000.dgi")
asrc = r'[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc1/BDMV/STREAM/00000.mka'

# # # # # # # # # #
# Cuts, variables #
# # # # # # # # # #
ac.eztrim(src,[(24,-24)], asrc,"fate_go00.mka")
src = src[24:-24]
b, c = 0, 1/2
desc_w, desc_h = 1551, 872.5
y_shift = (src.height/desc_h)*0.25

# # # # # # # # #
# Chroma shift  #
# # # # # # # # #
planes = kgf.split(fvf.Depth(src, 32))
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
shift = kgf.join(planes)

# # # # # # # # # # # # # # # # # # # # # # # #
# Denoise with CoolDegrain in single floating #
# # # # # # # # # # # # # # # # # # # # # # # #
den = CoolDegrainSF(shift, tr=2, thsad=60, blksize=8, overlap=4)

# # # # # # # # # # # #
# Descale 1551x872.5  #
# # # # # # # # # # # #
def gradient_function(w, h):
	blank = core.std.BlankClip(format=vs.GRAYS, length=1, width=w, height=1)
	stripes = []
	black = 0
	color = 1
	while color >= black:
		stripes.append(core.std.BlankClip(blank, color=color))
		color -= 1 / (ceil(h) / 2)
	gradient = core.std.StackVertical(stripes)
	black = core.std.BlankClip(format=vs.GRAYS, length=1, width=w, height=ceil(h)/2)
	return core.std.StackVertical([gradient, black])

# Mask top and bottom
mask_top = gradient_function(desc_w, desc_h)
mask_top = core.resize.Spline36(mask_top, src_top=-0.25)
mask_bot = mask_top.std.FlipVertical()
mask_bot = core.resize.Spline36(mask_bot, src_top=0.25)
# Descale clips
y = kgf.get_y(den)
desc = 		core.descale.Debicubic(y, desc_w, ceil(desc_h), b, c, cache_size=5)
desc_top = 	core.descale.Debicubic(y, desc_w, ceil(desc_h), b, c, src_top=-y_shift, cache_size=5)
desc_bot = 	core.descale.Debicubic(y, desc_w, ceil(desc_h), b, c, src_top=y_shift, cache_size=5)
# Merging
desc = core.std.MaskedMerge(desc, desc_top, mask_top)
desc = core.std.MaskedMerge(desc, desc_bot, mask_bot)

# # # # # #
# Rescale #
# # # # # #
# Upscaled_sraa of lvsfunc with Repair in single floating
scaled = mdf.upscaled_sraa_mod(desc, 2, 20, src.height, 16/9, False)
# Mask top and bottom
mask_top = gradient_function(src.width, src.height)
mask_bot = mask_top.std.FlipVertical()
# Upscale clips shifted
scaled_top = core.resize.Spline36(scaled, src_top=-y_shift)
scaled_bot = core.resize.Spline36(scaled, src_top=y_shift)
# Merging
scaled = core.std.MaskedMerge(scaled, scaled_top, mask_top)
scaled = core.std.MaskedMerge(scaled, scaled_bot, mask_bot)
# Merge the chroma back
scaled = core.std.ShufflePlanes([scaled, den], [0, 1, 2], vs.YUV)
scaled = fvf.Depth(scaled, 16)

# # # # # # #
# Debanding #
# # # # # # #
# Same as retinex_edgemask of kagefunc but I use tcannyCL here
db_m = mdf.retinex_edgemask_mod(scaled, opencl=True)
db = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, scaled, db_m)

# # # # # # # #
# Regraining  #
# # # # # # # #
grain_a = kgf.adaptive_grain(db)
grain_b = core.grain.Add(db, 3, constant=True)
crop = core.std.CropRel(grain_b, left=0, top=166, right=0, bottom=166)
bd = core.std.AddBorders(crop, left=0, top=166, right=0, bottom=166)
grain = fvf.rfs(grain_a, bd, mappings="[21535 22178]")

# # # # # # # # #
# 1080p things  #
# # # # # # # # #
ref = fvf.Depth(den, 16)
credit_m = vrdf.DRM(ref, desc_h, 'bicubic', b, c).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, ref, credit_m[182], 0), mappings='[119 206]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[331], 0), mappings='[256 355]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, vrdf.RegionMask(credit_m[523], left=0, right=745, top=0, bottom=0), 0), mappings='[468 547]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[790], 0), mappings='[731 814]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[1038], 0), mappings='[970 1053]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[1184], 0), mappings='[1125 1208]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[1337], 0), mappings='[1287 1361]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[1486], 0), mappings='[1439 1510]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[2302], 0), mappings='[2243 2326]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[2476], 0), mappings='[2413 2500]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[2637], 0), mappings='[2599 2682]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[2786], 0), mappings='[2727 2810]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[3476], 0), mappings='[3394 3476]')

credit = fvf.rfs(credit, ref, mappings='[35941 38109]')

# # # # # # # # # # # # # #
# Final dither and output #
# # # # # # # # # # # # # #
final = fvf.Depth(credit, 10)

final.set_output()