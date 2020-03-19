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
src = 		lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00008.m2ts')
src_prev = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00014.m2ts')
encard = 	lvf.src(r'episode-2-just-the-slightest-bit-of-motivation-733625 [00_23_42.546] 1.png')
encard = core.std.AssumeFPS(encard, fpsnum=src.fps.numerator, fpsden=src.fps.denominator)
encard = core.resize.Spline36(encard, format=src.format, matrix_s=mvf.GetMatrix(src, dIsRGB=True))

# Audio cuts
asrc = 		r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00008.mka'
asrc_prev = r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00014.mka'
ablank = 	r'blank.wav'
ac.eztrim(src,	[(24,34070)], asrc, 	'rokuaka02a.mka')
ac.eztrim(src,	[(24,744)], asrc_prev, 	'rokuaka02b.mka')
ac.eztrim(src,	[(0,239)], 	ablank, 	'rokuaka02c.mka')

# Video cuts
src = src[24:34070]
src_prev = src_prev[24:744]

# 0:24:20.000
src = src+src_prev+(encard*239)

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'rokuaka02_keyframes.log')

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

# Rescale & merge
upsc_y = vrdf.Bicubic(desc_y, 1920, 1080, 'robidoux sharp')
arte_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80, sw=0, sh=0)
arte_m = kgf.iterate(arte_m, core.std.Inflate, 2)

scaled_y = znedi3_rpow2(den_y, 2, nns=4).resize.Spline36(1920, 1080)
scaled_y = core.std.MaskedMerge(scaled_y, get_y(src16), arte_m)

merge = core.std.ShufflePlanes([scaled_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Extra aa to kill remaining aliasing
line_m = kgf.retinex_edgemask(merge)
aa = taa.TAAmbk(merge, aatype='Nnedi3', nsize=4, nns=4, mclip=line_m, opencl=True)

# Deband
db_a = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 0, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 18, 80, 80, 80, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, aa, line_m)
db = fvf.rfs(db, db_b, mappings="[31252 31631]")

# Random 1080p things
credit_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80).std.Inflate()
credit = fvf.rfs(db, core.std.MaskedMerge(db, src16, vrdf.RegionMask(credit_m[118], left=0, right=0, top=0, bottom=193)), mappings='[59 142]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[373], left=1318, right=0, top=0, bottom=388)), mappings='[321 428]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[373], left=1186, right=0, top=0, bottom=753)), mappings='[471 584]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[771], left=1000, right=0, top=0, bottom=0)), mappings='[693 800]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[1143], left=0, right=1055, top=0, bottom=0)), mappings='[1048 1167]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[1507], left=0, right=0, top=0, bottom=697)), mappings='[1481 1588]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[1672], left=0, right=0, top=0, bottom=724)), mappings='[1613 1720]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[1804], left=0, right=0, top=0, bottom=665)), mappings='[1745 1852]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[2228], left=0, right=653, top=482, bottom=0)), mappings='[2178 2285]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[2678], left=860, right=0, top=0, bottom=0)), mappings='[2649 2726]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[2812], left=1204, right=0, top=0, bottom=0)), mappings='[2781 2924]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[2989]), mappings='[2961 3080]')

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[3339]), mappings='[3333 3389]')

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32083]), mappings='[32069 32152]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32199]), mappings='[32165 32259]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32336]), mappings='[32279 32362]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32405]), mappings='[32405 32488]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[32789], left=0, right=0, top=426, bottom=0)), mappings='[32766 32847]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32903]), mappings='[32860 32951]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33240], left=1147, right=0, top=0, bottom=0)), mappings='[33179 33286]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33318], left=1251, right=0, top=0, bottom=0)), mappings='[33306 33401]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33511], left=0, right=143, top=645, bottom=0)), mappings='[33438 33538]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[33643]), mappings='[33575 33643]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[33835]), mappings='[33746 33835]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[33844]), mappings='[33836 33913]')

# Galaxy brain filtering for the endcard
endcard = vrdf.Bicubic(core.std.Crop(credit, 2, 2, 0, 0), 1920, 1080, 'robidoux')
endcard = vrdf.KNLMCL(endcard, 1.5, 1.5, depth=16)
endcard = hvf.HQDeringmod(endcard, mrad=1, msmooth=1, mthr=80, minp=1, drrep=13, thr=24, planes=[0, 1, 2], show=False)
endcard = lvf.upscaled_sraa(endcard)
endcard = core.neo_f3kdb.Deband(endcard, 17, 36, 36, 36, 24, 0, sample_mode=4)
endcard = fvf.rfs(credit, endcard, mappings="[34766 35004]")

final = fvf.Depth(endcard, 10)

final.set_output()