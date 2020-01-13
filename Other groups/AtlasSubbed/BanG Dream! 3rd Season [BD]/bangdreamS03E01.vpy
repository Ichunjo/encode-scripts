import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import descale as dsc
from vsutil import iterate, join, split

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src = lvf.src(r"BRMM-10222BD/BDMV/STREAM/00000.m2ts")
asrc = r"BRMM-10222BD/BDMV/STREAM/00000.mka"
ac.eztrim(src, [(0,34047)], asrc, "bangdreamS03E01.mka")

src = src[:34046]

# Keyframes
vrdf.generate_keyframes(src, 'bangdreamS03E01_keyframes.log')

# Variables
opstart, opend = 1822, 3980
edstart, edend = 31528, 33685
flashback = '[6401 6574] [15004 15330] [21341 21537]'

# Dither
src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

# Chroma shift
planes = split(src32)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
uv_shift = join(planes)

# Descale
desc_y = dsc.Descale(src32, 1433, 806, 'bilinear', gray=True)

# Denoise luma & chroma
den_y = core.knlm.KNLMeansCL(desc_y, a=2, h=0.55, d=3, device_type='gpu')
den_uv = core.knlm.KNLMeansCL(uv_shift, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

# AA & upscale
scaled_y = lvf.upscaled_sraa(den_y, 2, None, 1080, False)

# Merged
merged = core.std.ShufflePlanes([scaled_y, den_uv], [0, 1, 2], vs.YUV)
merged = fvf.Depth(merged, 16)

# Custom deband mask
db_mask = kgf.retinex_edgemask(merged)
for i in range(0, 3):
    db_mask = db_mask.rgvs.RemoveGrain(3)
db_mask = iterate(db_mask, core.std.Inflate, 3)

# Deband
db = core.neo_f3kdb.Deband(merged, 17, 36, 36, 36, 28, 0, sample_mode=4)
db = core.std.MaskedMerge(db, merged, db_mask)

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.4, static=False, luma_scaling=8)
grain_c = kgf.adaptive_grain(grain_a, 0.6, static=False)
grain = fvf.rfs(grain_a, grain_b, mappings=flashback)
grain = fvf.rfs(grain_a, grain_c, mappings='[28117 31527]')

# 1080p things
c_mask = vrdf.DRM(src16, 806, 'bilinear', mthr=80, sw=2, sh=2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, c_mask), mappings=f'[{opstart+460} {opstart+524}]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[4045]), mappings='[3993 4093]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[33964]), mappings='[33895 33964]')

# Just a mild aa since the ED seems 1080p native
aa_ed = taa.TAAmbk(src16, aatype='Nnedi3', nsize=4, nns=4, mytpe=1, down8=False, opencl=True)
aa_ed = fvf.rfs(credit, aa_ed, mappings=f'[{edstart} {edend}]')

final = fvf.Depth(aa_ed, 10)

final.set_output()