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
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

src = core.dgdecodenv.DGSource(r"BRMM_10241BD/STREAM/00001.dgi")
asrc = r"BRMM_10241BD/STREAM/00001.wav"
eztrim(src, (0, 34047), asrc, "bangdreamS03E08.mka")

src = src[:34047]

# Variables
opstart, opend = 2206, 4362
edstart, edend = 31529, 33686
flashback = '[6391 7061] [26078 26227]'

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
db = core.std.MaskedMerge(db_a, merged, line_m)

# Grain
grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.4, static=False, luma_scaling=8)
grain = fvf.rfs(grain_a, grain_b, mappings=flashback)

# 1080p things
c_mask = vrf.drm(src16, 806, 'bilinear', mthr=80, sw=2, sh=2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, c_mask), mappings=f'[{opstart+460} {opstart+524}]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask), mappings='[4504 4616]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, c_mask[33967]), mappings='[33912 33968]')


# src a mild aa since the ED seems 1080p native
aa_ed = TAAmbk(src16, aatype='Nnedi3', nsize=4, nns=4, mytpe=1, down8=False, opencl=True)
aa_ed = fvf.rfs(credit, aa_ed, mappings=f'[{edstart} {edend}]')

final = fvf.Depth(aa_ed, 10)

final.set_output()