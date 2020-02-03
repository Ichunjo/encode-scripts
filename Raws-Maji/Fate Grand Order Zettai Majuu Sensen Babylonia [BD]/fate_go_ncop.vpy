import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import descale as dsc
import vardefunc as vrdf
import modfunc as mdf
import debandshit as dbs
import mvsfunc as mvf
import acsuite
from cooldegrain import CoolDegrainSF
from insaneAA import rescale

core = vs.core
core.max_cache_size = 26000
ac = acsuite.AC()

# For some reasons I can't create a blank clip in GRAY10.
# The goal here is to create a gradient mask to gradually shift the clip 
# and simulate a shrinking.
def gradient_m(clip):
	blank = core.std.BlankClip(format=vs.GRAY16, length=1, width=clip.width, height=1)
	blank = fvf.Depth(blank, 10)
	maxval = 1023
	color = 0
	stripes = []
	while color <= maxval:
		stripes.append(core.std.BlankClip(blank, color=color))
		color += 1
	gradientMask = core.std.StackVertical(stripes)
	return core.resize.Point(gradientMask, clip.width, clip.height)

src = core.dgdecodenv.DGSource(r"[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc2/BDMV/STREAM/00007.dgi")

# # # # # # # # # #
# Cuts, variables #
# # # # # # # # # #
asrc = r'[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc2/BDMV/STREAM/00007.mka'
ac.eztrim(src,[(24,-24)], asrc,"fate_go_ncop.mka")

src = src[24:-24]
desc_str = 0.5
opstart = 0
opening_a = f'[{opstart+378} {opstart+401}] [{opstart+1728} {opstart+1757}]'
opening_b = f'[{opstart+1636} {opstart+1673}]'

# # # # # # # # #
# Chroma shift  #
# # # # # # # # #
planes = kgf.split(fvf.Depth(src, 32))
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
shift = kgf.join(planes)

# # # # # # # # # # # # # # # # # # # # # # # #
# Denoise with CoolDegrain in single floating #
# # # # # # # # # # # # # # # # # # # # # # # #
adapt_m = vrdf.adaptive_mask(shift, 60)
den_a = CoolDegrainSF(shift, tr=1, thsad=48, blksize=8, overlap=4)
den_b = CoolDegrainSF(shift, tr=5, thsad=60, blksize=8, overlap=4)
den_c = mvf.BM3D(shift, [3, 2], radius1=1)
den_c = core.std.MaskedMerge(den_b, den_c, adapt_m)
den = fvf.rfs(den_a, den_b, mappings=opening_a)
den = fvf.rfs(den, den_c, mappings=opening_b)

# # # # # # # # # # # #
# Descale 1551x872.5  #
# # # # # # # # # # # #
y, u, v = kgf.split(den)

# Combined descale / downscale to avoid ringing artefacts
desc = dsc.Descale(y, 1551, 873, 'bicubic', 0, 1/2)
down = core.resize.Spline36(y, 1551, 873)
desc = core.std.Expr([desc, down], f'x {desc_str} * y 1 {desc_str} - * +')

# Slight shift to transform the height 873 -> 872.5 
desc_shift = core.resize.Spline36(desc, src_top=0.5)

# And merge with the gradient mask.
desc_good = core.std.MaskedMerge(desc, desc_shift, fvf.Depth(gradient_m(desc), 32))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Rescale by using a part of insaneAA which is much better in this case #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
scaled = rescale(desc, fasterAA=True, nnedi3Mode=('nnedi3', 'opencl'), dx=1920, dy=1080, alpha=0.25, beta=0.3, gamma=500)
scaled = kgf.join([scaled, u, v])

# Some artefacts are too obvious and I prefer not descale them
artefact = [
	vrdf.DRM(y, 872, 'bicubic', 0, 1/2, mthr=40, sw=0, sh=0),
	vrdf.DRM(y, 873, 'bicubic', 0, 1/2, mthr=40, sw=0, sh=0),
]

# # # # # # # # #
# Anti-aliasing #
# # # # # # # # #
aa_m = core.std.Expr(artefact, 'x y +')
# But some of them are aliased so let's AA them
aa = lvf.transpose_aa(fvf.Depth(den, 16))
aa = core.std.MaskedMerge(fvf.Depth(scaled, 16), aa, fvf.Depth(aa_m, 16), 0)

# # # # # # #
# Debanding #
# # # # # # #
# Same as retinex_edgemask of kagefunc but I use tcannyCL here
db_m = mdf.retinex_edgemask_mod(aa, opencl=True).rgvs.RemoveGrain(4)
db_a = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, aa, db_m)
db_b = dbs.f3kbilateral(aa, 15, 48, 48)
db_c = dbs.f3kbilateral(aa, 18, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=32, preset='depth')
db_c = core.std.MaskedMerge(db_b, db_c, fvf.Depth(adapt_m, 16))
db = fvf.rfs(db, db_b, mappings=opening_a)
db = fvf.rfs(db, db_c, mappings=opening_b)

# # # # # # # #
# Regraining  #
# # # # # # # #
grain_a = kgf.adaptive_grain(db)
grain_b = kgf.adaptive_grain(db, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=opening_b)

# # # # # # # # # # # # # #
# Final dither and output #
# # # # # # # # # # # # # #
final = fvf.Depth(grain, 10)

final.set_output()