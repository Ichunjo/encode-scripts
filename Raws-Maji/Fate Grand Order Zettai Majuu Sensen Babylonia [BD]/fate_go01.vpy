import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import acsuite
import psutil
from descale import Descale
from cooldegrain import CoolDegrainSF
from insaneAA import rescale
p_handle = psutil.Process()
p_handle.cpu_affinity(list(range(0,23,2)))

core = vs.core
core.max_cache_size = 26000
core.num_threads = 12

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

src = core.dgdecodenv.DGSource(r"[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc1/BDMV/STREAM/00001.dgi")

# # # # # # # # # #
# Cuts, variables #
# # # # # # # # # #
asrc = r'[BDMV] Fate Grand Order - Absolute Demonic Front Babylonia [Vol.1] [JP]/Disc1/BDMV/STREAM/00001.mka'
ac.eztrim(src,[(24,-24)], asrc,"fate_go01.mka")

src = src[24:-24]
desc_str = 0.5

# # # # # # # # #
# Chroma shift  #
# # # # # # # # #
planes = kgf.split(fvf.Depth(src, 32))
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
shift = kgf.join(planes)

# # # # # # # # # # # # # # # # # # # # # # # #
# Denoise with CoolDegrain in single floating #
# # # # # # # # # # # # # # # # # # # # # # # #
den = CoolDegrainSF(shift, tr=1, thsad=48, blksize=8, overlap=4)

# # # # # # # # # # # #
# Descale 1551x872.5  #
# # # # # # # # # # # #
y, u, v = kgf.split(den)

# Combined descale / downscale to avoid ringing artefacts
desc = Descale(y, 1551, 873, 'bicubic', 0, 1/2)
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
db = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, db_m)

# # # # # # # #
# Regraining  #
# # # # # # # #
grain_a = kgf.adaptive_grain(db)
grain_b = core.grain.Add(db, 2, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings="[5360 5683]")

# # # # # # # # #
# 1080p things  #
# # # # # # # # #
ref = fvf.Depth(den, 16)
credit_m = vrdf.DRM(ref, 873, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, ref, credit_m, 0), mappings='[215 2283]')

credit = fvf.rfs(credit, ref, mappings='[31338 33506]')

# # # # # # # # # # # # # #
# Final dither and output #
# # # # # # # # # # # # # #
final = fvf.Depth(credit, 10)
final.set_output()