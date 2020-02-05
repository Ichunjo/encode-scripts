import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import acsuite
from cooldegrain import CoolDegrainSF
from math import ceil

core = vs.core
core.max_cache_size = 26000

ac = acsuite.AC()

src = core.dgdecodenv.DGSource(r"[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc2/BDMV/STREAM/00008.dgi")
asrc = r'[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc2/BDMV/STREAM/00008.mka'

# # # # # # # # # #
# Cuts, variables #
# # # # # # # # # #
ac.eztrim(src,[(24,-28)], asrc,"fate_go_nced.mka")
src = src[24:-28]
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
den = CoolDegrainSF(shift, tr=2, thsad=60, blksize=8, overlap=4, plane=0)
den = core.knlm.KNLMeansCL(den, a=2, h=0.3, d=3, device_type='gpu', channels='UV')

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
db_a = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, scaled, db_m)

# # # # # # # #
# Regraining  #
# # # # # # # #
grain = kgf.adaptive_grain(db)

# # # # # # # # # # # # # #
# Final dither and output #
# # # # # # # # # # # # # #
final = fvf.Depth(grain, 10)

final.set_output()