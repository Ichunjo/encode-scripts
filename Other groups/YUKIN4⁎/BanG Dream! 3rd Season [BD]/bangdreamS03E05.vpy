import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
import vsTAAmbk as taa
import descale as dsc
import debandshit as dbs
from vsutil import iterate, join, split
from cooldegrain import CoolDegrain
from edi_rpow2 import znedi3_rpow2

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src = core.dgdecodenv.DGSource(r"BRMM-10228BD/BDMV/STREAM/00000.dgi")
asrc = r"BRMM-10228BD/BDMV/STREAM/00000.mka"
ac.eztrim(src, [(0,34047)], asrc, "bangdreamS03E05.mka")

src = src[:34047]

# Keyframes
vrdf.generate_keyframes(src, 'bangdreamS03E05_keyframes.log')

# Variables
opstart, opend = 2925, 5082
edstart, edend = 31529, 33686
flashback = '[9510 9564] [9906 10094] [13504 13872]'
live = 		'[24581 24701] [24748 24817] [24924 25061] [25197 25579] [25640 25687] [33924 34046]'

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
den_y = CoolDegrain(desc_y, tr=1, thsad=60, thsadc=0, bits=16, blksize=8, overlap=4)

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
db_b = dbs.f3kbilateral(merged, 20, 56, 48)
db_b = core.neo_f3kdb.Deband(db_b, grainy=32, preset='depth')
db = core.std.MaskedMerge(db_a, merged, line_m)
db = fvf.rfs(db, db_b, mappings='[24813 24817] [24954 24974] [25197 25269]')

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.4, static=False, luma_scaling=8)
grain_c = kgf.adaptive_grain(grain_a, 0.8, static=False)
grain = fvf.rfs(grain_a, grain_b, mappings=flashback)
grain = fvf.rfs(grain, grain_c, mappings=live)

# 1080p things
c_mask = vrdf.DRM(src16, 806, 'bilinear', mthr=80, sw=2, sh=2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, c_mask), mappings=f'[{opstart+460} {opstart+524}]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[5434]), mappings='[5371 5458]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[33899]), mappings='[33899 33964]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask), mappings='[23284 26232]')

# src a mild aa since the ED seems 1080p native
aa_ed = taa.TAAmbk(src16, aatype='Nnedi3', nsize=4, nns=4, mytpe=1, down8=False, opencl=True)
aa_ed = fvf.rfs(credit, aa_ed, mappings=f'[{edstart} {edend}]')

final = fvf.Depth(aa_ed, 10)

final.set_output()