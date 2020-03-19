import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import acsuite
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 8192

ac = acsuite.AC()

# Index
src = lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 3/BDMV/STREAM/00009.m2ts')

# Audio cuts
asrc = r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 3/BDMV/STREAM/00009.mka'
ac.eztrim(src,	[(24,-24)], asrc, 	'rokuaka_nced.mka')

# Video cuts
src = src[24:-24]

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'rokuaka_nced_keyframes.log')

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Adaptive denoise + simple chroma denoise
den_m = vrdf.adaptive_mask(src16, 20)
den_a = CoolDegrain(src16, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(src16, tr=1, thsad=24, thsadc=0, bits=16, blksize=8, overlap=4)
den_y = core.std.MaskedMerge(den_a, den_b, den_m)

den_uv = core.knlm.KNLMeansCL(src32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

merge = core.std.ShufflePlanes([den_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Extra aa to kill remaining aliasing
line_m = kgf.retinex_edgemask(merge)
aa = taa.TAAmbk(merge, aatype='Nnedi3', nsize=4, nns=4, mclip=line_m, opencl=True)

# Deband
db_a = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 0, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 64, 64, 64, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, aa, line_m)
db = fvf.rfs(db, db_b, mappings='[1413 1787]')

final = fvf.Depth(db, 10)

final.set_output()