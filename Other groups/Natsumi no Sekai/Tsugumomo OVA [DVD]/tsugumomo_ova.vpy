import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import modfunc as mdf
import muvsfunc as muvf
import fvsfunc as fvf
import vardefunc as vrdf
import debandshit as dbs
from  nnedi3_rpow2CL import nnedi3_rpow2CL
import acsuite

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = core.dgdecodenv.DGSource(r'[DVD][200122] Tsugumomo OVA/TSUGUMOMO_OVA/VIDEO_TS/VTS_01_1.dgi')
ncop = core.dgdecodenv.DGSource(r'00005.dgi')
asrc = r'[DVD][200122] Tsugumomo OVA/TSUGUMOMO_OVA/VIDEO_TS/VTS_01_1 T80 stereo 384 kbps DELAY 0 ms.ac3'
ancop = r'00005.mka'
ac.eztrim(src,[(0,-24)], asrc,"tsugumomo_ova.mka")
ac.eztrim(ncop,[(24,-24)], ancop,"ncop.mka")

# 2:3 pulldown so AAABB 
# A: Good frames B: Combed frames
detele = fvf.JIVTC(src, 1, tff=True)
detele = kgf.insert_clip(detele, detele[20097], 20098)
detele = fvf.Depth(detele, 32)[:-24]

den = mdf.hybriddenoise_mod(detele, knl=0.8, sigma=2, depth=32)

# Good 16/9 aspect ratio.
# Spline36 is sharper but introduce more ringing,
# The mask is here to avoid that.
ar_a = core.resize.Spline36(den, 864, 486)
ar_b = vrdf.Bicubic(den, 864, 486, 'robidoux sharp')
mask = muvf.AnimeMask(fvf.Depth(ar_b, 16), mode=-1)
mask = mask.std.Binarize(8000).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Deflate().std.Deflate()
ar = core.std.MaskedMerge(ar_b, ar_a, fvf.Depth(mask, 32))

# Upscale because easier for the typeset
scaled = nnedi3_rpow2CL(ar, nns=4).resize.Spline36(1024, 576)
scaled = fvf.Depth(scaled, 16)

# Sharping vertical lines and few blur for horizontal lines
sharp = muvf.Sharpen(scaled, amountH=1, amountV=-0.25)
mask = muvf.AnimeMask(scaled, mode=-1)
mask = mask.std.Binarize(6000).std.Maximum().std.Inflate().std.Deflate().std.Deflate()
sharp = core.std.MaskedMerge(scaled, sharp, mask)

# Fuck DVDs
dehalo = hvf.FineDehalo(sharp)
dering = hvf.EdgeCleaner(dehalo, 30, rmode=1, smode=1)

# Hell chroma banding 
edge_m = mdf.retinex_edgemask_mod(dering, opencl=True).std.Binarize(9800)
edge_m = edge_m.rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate()
db = dbs.f3kbilateral(dering, 15, 36, 64)
db = core.std.MaskedMerge(db, dering, edge_m)
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')

grain = kgf.adaptive_grain(db, 0.3, luma_scaling=4)

# Filterchain NCOP
ncop = fvf.Depth(ncop, 32)[24:-24]

planes = kgf.split(ncop)
planes[0] = kgf.inverse_scale(planes[0], height=720, kernel='bicubic', b=1/3, c=1/3, mask_detail=True)
planes[0] = lvf.upscaled_sraa(planes[0], 1.5, None, 576, 16/9, False)
planes[1], planes[2] = [
    core.resize.Spline36(plane, 1024/2, 576/2, src_left=vrdf.GetChromaShift(1080, 576))
    for plane in planes[1:]
]

scaled = kgf.join(planes)

den = mdf.hybriddenoise_mod(scaled, knl=0.3, sigma=1.25)
db = core.neo_f3kdb.Deband(den, 14, 36, 30, 30, 24, 0)

final = db + grain

final = fvf.Depth(final, 10)

final.set_output()