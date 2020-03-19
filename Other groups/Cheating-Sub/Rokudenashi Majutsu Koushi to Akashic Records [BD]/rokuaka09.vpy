import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
import vsTAAmbk as taa
import descale as dsc
import mvsfunc as mvf
import acsuite
from edi_rpow2 import znedi3_rpow2
from cooldegrain import CoolDegrain
from vsutil import get_w, get_y

core = vs.core
core.max_cache_size = 8192

ac = acsuite.AC()

# Index
src = 		lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 5/BDMV/STREAM/00007.m2ts')
src_prev = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 5/BDMV/STREAM/00009.m2ts')
src_ncop = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00009.m2ts')
src_nced = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 3/BDMV/STREAM/00009.m2ts')

encard = 	lvf.src(r'episode-9-the-reason-to-live-733639 [00_23_43.422] 1.png', src)

# Audio cuts
asrc = 		r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 5/BDMV/STREAM/00007.mka'
asrc_prev = r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 5/BDMV/STREAM/00009.mka'
ablank = 	r'blank.wav'
ac.eztrim(src,	[(24,34070)], asrc, 	'rokuaka09a.mka')
ac.eztrim(src,	[(24,1287)], asrc_prev, 	'rokuaka09b.mka')
ac.eztrim(src,	[(0,176)], 	ablank, 	'rokuaka09c.mka')

# Video cuts
src = src[24:34070]
src_prev = src_prev[24:1287]
src_ncop = src_ncop[24:-24]
src_nced = src_nced[24:-24]

# 0:24:40.020 :(
src = src + src_prev + encard[:176]

# Variables
opstart, opend = 1750, 3908

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'rokuaka09_keyframes.log')

# Dithers
src16 = fvf.Depth(src, 16)
src32 = fvf.Depth(src, 32)

# Descale
desc_y = dsc.Descale(src16, get_w(837), 837, 'bicubic', 0.2620, 0.3690, gray=True)

# Adaptive denoise + simple chroma denoise
den_m = vrdf.adaptive_mask(desc_y, 20)
den_a = CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(desc_y, tr=1, thsad=24, thsadc=0, bits=16, blksize=8, overlap=4)
den_y = core.std.MaskedMerge(den_a, den_b, den_m)

den_uv = core.knlm.KNLMeansCL(src32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

# Rescale
upsc_y = vrdf.Bicubic(desc_y, 1920, 1080, 'robidoux sharp')
arte_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80, sw=0, sh=0)
arte_m = kgf.iterate(arte_m, core.std.Inflate, 2)

scaled_y = znedi3_rpow2(den_y, 2, nns=4).resize.Spline36(1920, 1080)
scaled_y = core.std.MaskedMerge(scaled_y, get_y(src16), arte_m)

# Merging 
merge = core.std.ShufflePlanes([scaled_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Extra aa to kill remaining aliasing
line_m = kgf.retinex_edgemask(merge)
aa = taa.TAAmbk(merge, aatype='Nnedi3', nsize=4, nns=4, mclip=line_m, opencl=True)

# Deband
db = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

# Random 1080p things
credit_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80).std.Inflate()
credit = fvf.rfs(db, core.std.MaskedMerge(db, src16, credit_m[3910]), mappings='[3910 3980]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m), mappings='[32660 34014]')

# Opening credit
credit_op_m = vrdf.DCM(credit, src[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 4, 4)
credit_op = core.std.MaskedMerge(credit, src16, credit_op_m)

# Galaxy brain filtering for the endcard
endcard = vrdf.Bicubic(core.std.Crop(src16, 2, 2, 0, 0), 1920, 1080, 'robidoux')
endcard = vrdf.KNLMCL(endcard, 1, 1, depth=16)
endcard = lvf.upscaled_sraa(endcard, 2)
endcard = hvf.HQDeringmod(endcard, mrad=1, msmooth=1, mthr=80, minp=1, drrep=13, thr=24, planes=[0, 1, 2], show=False)
endcard = core.neo_f3kdb.Deband(endcard, 17, 36, 36, 36, 24, 0, sample_mode=4)
endcard = fvf.rfs(credit_op, endcard, mappings="[35309 35484]")

final = fvf.Depth(endcard, 10)

final.set_output()