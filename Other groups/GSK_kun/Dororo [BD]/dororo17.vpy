import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import vsTAAmbk as taa
import vardefunc as vrdf
import descale as dsc
import debandshit as dbs
import acsuite
from edi_rpow2 import znedi3_rpow2
from cooldegrain import CoolDegrain
from vsutil import get_w

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r'[BDMV] どろろ Blu-ray BOX/下巻/DORORO_3/BDMV/STREAM/00005.m2ts')

asrc = r'[BDMV] どろろ Blu-ray BOX/下巻/DORORO_3/BDMV/STREAM/00005.mka'
ac.eztrim(src,[(24,34190)], asrc, "dororo17.mka")

src = src[24:34190]

# Keyframes for x264 --qpfile 
vrdf.generate_keyframes(src, 'dororo17_keyframes.log')

# Frame variables
opstart, opend = 4468, 6617 # opstart: first frame with light
epnamestart, epnameend = opend+1, 6713
edstart, edend = 31888, 34045
cardstart, cardend = edend+1, src.num_frames-1

# Zones
opening_a = f'[{opstart} {opstart+741}]'
opening_b = f'[{opstart+742} {opend}]'
opening_c = f'[{opstart+1182} {opstart+1365}]'
ending = f'[{edstart} {edend}]'

# Dither
src = fvf.Depth(src, 32)

# Chroma shift
planes = kgf.split(src)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
uv_shift32 = kgf.join(planes)
uv_shift16 = fvf.Depth(src, 16)

# Descale
desc = dsc.Descale(uv_shift16, get_w(806), 806, 'bicubic', 0, 1/2, gray=True)

# Luma noise stabilization
den_clip = core.knlm.KNLMeansCL(desc, a=2, h=0.35, d=1, device_type='gpu')
stab = hvf.GSMC(desc, den_clip, radius=3)

# Luma adaptive denoise for episodes
adapt_m = vrdf.adaptive_mask(desc, 30)
den_y_a = CoolDegrain(stab, tr=1, thsad=72, thsadc=0, bits=16, blksize=8, overlap=4)
den_y_b = CoolDegrain(stab, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_y = core.std.MaskedMerge(den_y_a, den_y_b, adapt_m)

# Just a standard chroma denoise
den_uv = core.knlm.KNLMeansCL(uv_shift32, a=2, h=0.6, d=3, device_type='gpu', channels='UV')

# Rescaling & merge luma/chroma
scaled_y = vrdf.Bicubic(znedi3_rpow2(den_y, 2, nns=4, qual=2), 1920, 1080, 'catrom')
scaled = core.std.ShufflePlanes([scaled_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Luma & chroma denoise without rescaling. Opening parts
den_a = CoolDegrain(uv_shift16, tr=1, thsad=24, thsadc=48, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(uv_shift16, tr=1, thsad=48, thsadc=48, bits=16, blksize=8, overlap=4)

scaled = fvf.rfs(scaled, den_a, mappings=opening_a)
scaled = fvf.rfs(scaled, den_b, mappings=ending)

# Deband masks
scaled8 = fvf.Depth(scaled, 8)
line_m_a = taa.TAAmbk(scaled8, mtype=1, showmask=1).std.Inflate()
line_m_b = kgf.retinex_edgemask(scaled8).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate().std.Inflate()
line_m = core.std.Expr([line_m_a, line_m_b], 'x y +')
line_m = core.std.Expr(line_m, 'x 30 < 0 x 3 * ?').std.Inflate().std.Inflate()
adapt_m = vrdf.adaptive_mask(scaled, 60)

# Grain & scenefiltering
db_a = core.neo_f3kdb.Deband(scaled, 17, 42, 48, 48, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(scaled, 14, 48, 54, 54, 32, 0, sample_mode=4)
db_c = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db_d = dbs.f3kbilateral(scaled, 20, 64, 64)
db_d = core.grain.Add(db_d, 0.1)

db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = core.std.MaskedMerge(db, scaled, fvf.Depth(line_m, 16))

db = fvf.rfs(db, db_c, mappings=ending)
db = fvf.rfs(db, db_d, mappings=opening_c)

# Grain
grain = kgf.adaptive_grain(db, 0.25)

# Episode name & endcard
aa_b = lvf.upscaled_sraa(uv_shift16, 2, rep=13, sharp_downscale=False)
tag = fvf.rfs(grain, aa_b, mappings=f'[{epnamestart} {epnameend}] [{cardstart} {cardend}]')
tag = core.std.FreezeFrames(tag, epnamestart, epnameend, epnameend)
tag = core.std.FreezeFrames(tag, cardstart, cardend, cardend)

# 1080p elements
credit_m = vrdf.DRM(uv_shift16, 806, 'bicubic', 0, 1/2).std.Inflate()
credit_op = fvf.rfs(tag, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+747], 0), mappings=f'[{opstart+742} {opstart+835}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+929], 0), mappings=f'[{opstart+929} {opstart+1014}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+1182], 0), mappings=f'[{opstart+1182} {opstart+1277}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+1309], 0), mappings=f'[{opstart+1295} {opstart+1365}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+1548], 0), mappings=f'[{opstart+1493} {opstart+1596}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+1640], 0), mappings=f'[{opstart+1620} {opstart+1703}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+1734], 0), mappings=f'[{opstart+1712} {opstart+1799}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+2052], 0), mappings=f'[{opstart+1998} {opstart+2100}]')
credit_op = fvf.rfs(credit_op, core.std.MaskedMerge(tag, uv_shift16, credit_m[opstart+2101], 0), mappings=f'[{opstart+2101} {opend}]')

# Dither final
final = fvf.Depth(credit_op, 10)

final.set_output()