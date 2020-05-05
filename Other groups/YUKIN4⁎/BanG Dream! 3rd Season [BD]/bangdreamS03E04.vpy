from vsutil import iterate, join, split
from cooldegrain import CoolDegrain
from edi_rpow2 import znedi3_rpow2
from vsTAAmbk import TAAmbk
from acsuite import eztrim
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrf
import descale as dsc
import debandshit as dbs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

src = core.dgdecodenv.DGSource(r"BRMM-10228BD/BDMV/STREAM/00001.dgi")
asrc = r"BRMM-10228BD/BDMV/STREAM/00001.mka"
eztrim(src, (0, 34048), asrc, "bangdreamS03E04.mka")

src = src[:34048]

# Variables
opstart, opend = 4915, 7072
edstart, edend = 31530, 33687
flashback = '[456 703] [7351 7689] [22974 23100] [23678 23785]'
live = '[7351 7689] [25817 28238]'

# Dither
src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

# Chroma shift
planes = split(src32)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
uv_shift = join(planes)

# Descale
desc_y = dsc.Descale(src16, 1432, 806, 'bilinear', gray=True)

# Denoise luma & chroma
den_y_a = CoolDegrain(desc_y, tr=1, thsad=60, thsadc=0, bits=16, blksize=8, overlap=4)
den_y_b = core.knlm.KNLMeansCL(desc_y, a=2, h=1.2, d=3, device_type='gpu')
den_y = fvf.rfs(den_y_a, den_y_b, mappings='[28024 28079] [28199 28257]')

sharp_y = hvf.ContraSharpening(den_y, desc_y, 3, planes=0)

den_uv = core.knlm.KNLMeansCL(uv_shift, a=2, h=0.65, d=3, device_type='gpu', channels='UV')

# AA & upscale
line_m = kgf.retinex_edgemask(src16).std.Binarize(8500)
line_m = core.rgvs.RemoveGrain(line_m, 3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
line_m = iterate(line_m, core.std.Inflate, 3)

scaled_y_line = lvf.upscaled_sraa(sharp_y, 2, None, 1080, 16/9, False)
scaled_y_tex = znedi3_rpow2(sharp_y, 2, nsize=2).resize.Spline36(1920, 1080)
scaled_y = core.std.MaskedMerge(scaled_y_tex, scaled_y_line, line_m)

# Merged
merged = core.std.ShufflePlanes([scaled_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Déband with mask
db_a = core.neo_f3kdb.Deband(merged, 16, 36, 36, 36, 28, 0, sample_mode=4)
db_b = dbs.f3kbilateral(merged, 10, 64, 48)
db_b = dbs.f3kbilateral(db_b, 20, 48, 36)
db_b = core.neo_f3kdb.Deband(db_b, grainy=32, preset='depth')
db_c = core.neo_f3kdb.Deband(merged, 17, 64, 64, 64, 28, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)
db = fvf.rfs(db, db_b, mappings='[27801 27821]')
db = fvf.rfs(db, db_c, mappings='[28199 28257]')

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.4, static=False, luma_scaling=8)
grain_c = kgf.adaptive_grain(grain_a, 0.8, static=False)
grain_d = kgf.adaptive_grain(grain_a, 1.5, static=False, luma_scaling=16)
grain = fvf.rfs(grain_a, grain_b, mappings=flashback)
grain = fvf.rfs(grain, grain_c, mappings=live)
grain = fvf.rfs(grain, grain_d, mappings='[28024 28079]')

# 1080p things
c_mask = vrf.drm(src16, 806, 'bilinear', mthr=80, sw=2, sh=2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, c_mask), mappings=f'[{opstart+460} {opstart+524}]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[7213]), mappings='[7152 7237]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[33907]), mappings='[33907 33966]')

# Mild aa since the ED seems 1080p native
aa_ed = TAAmbk(src16, aatype='Nnedi3', nsize=4, nns=4, mytpe=1, down8=False, opencl=True)
aa_ed = fvf.rfs(credit, aa_ed, mappings=f'[{edstart} {edend}]')

final = fvf.Depth(aa_ed, 10)

final.set_output()