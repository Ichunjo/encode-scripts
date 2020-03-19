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
src = 		lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00007.m2ts')
src_prev = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00013.m2ts')
encard = 	lvf.src(r'episode-1-the-unmotivated-bastard-733623 [00_23_41.420] 1.png')
encard = core.std.AssumeFPS(encard, fpsnum=src.fps.numerator, fpsden=src.fps.denominator)
encard = core.resize.Spline36(encard, format=src.format, matrix_s=mvf.GetMatrix(src, dIsRGB=True))

# Audio cuts
asrc = 		r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00007.mka'
asrc_prev = r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 1/BDMV/STREAM/00013.mka'
ablank = 	r'blank.wav'
ac.eztrim(src,	[(24,-25)], asrc, 		'rokuaka01a.mka')
ac.eztrim(src,	[(24,768)], asrc_prev, 	'rokuaka01b.mka')
ac.eztrim(src,	[(0,214)], 	ablank, 	'rokuaka01c.mka')

# Video cuts
src = src[24:-25]
src_prev = src_prev[24:768]

# 0:24:20.000
src = src+src_prev+(encard*214)

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'rokuaka01_keyframes.log')

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
arte_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80, sw=0, sh=0)
arte_m = kgf.iterate(arte_m, core.std.Inflate, 2)

scaled_y = znedi3_rpow2(den_y, 2, nns=4).resize.Spline36(1920, 1080)
scaled_y = core.std.MaskedMerge(scaled_y, get_y(src16), arte_m)

merge = core.std.ShufflePlanes([scaled_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

# Extra aa to kill remaining aliasing
line_m = kgf.retinex_edgemask(merge)
aa = taa.TAAmbk(merge, aatype='Nnedi3', nsize=4, nns=4, mclip=line_m, opencl=True)

# Deband
db = core.neo_f3kdb.Deband(aa, 17, 30, 30, 30, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

# Random 1080p things
credit_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80).std.Inflate()
credit = fvf.rfs(db, core.std.MaskedMerge(db, src16, credit_m[2222]), mappings='[2169 2322]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[2501]), mappings='[2470 2611]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[2817]), mappings='[2726 2869]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[3181]), mappings='[3128 3259]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[3499]), mappings='[3409 3571]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[3774]), mappings='[3745 3886]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[4018]), mappings='[3923 4066]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[4226]), mappings='[4101 4250]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[4405]), mappings='[4286 4453]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[4745]), mappings='[4650 4793]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[5028]), mappings='[4964 5124]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[5193]), mappings='[5171 5308]')

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[31714]), mappings='[31688 31758]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[31800], 0, 617, 0, 0)), mappings='[31771 31887]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[31945], 804, 0, 527, 0)), mappings='[31912 32017]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[32329], 544, 0, 0, 609)), mappings='[32284 32362]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[32477], 0, 789, 0, 448)), mappings='[32381 32477]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32766]), mappings='[32757 32851]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[32948]), mappings='[32876 32983]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33203], 683, 0, 0, 0)), mappings='[33140 33246]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33378], 0, 0, 0, 509)), mappings='[33336 33419]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33455], 0, 0, 0, 509)), mappings='[33426 33503]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[33675]), mappings='[33599 33675]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33777], 1117, 0, 0, 0)), mappings='[33749 33836]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[33926], 1038, 0, 0, 0)), mappings='[33855 33926]')

credit = fvf.rfs(credit, src16, mappings='[33927 34046]')

# Galaxy brain filtering for the endcard
endcard = vrdf.Bicubic(core.std.Crop(credit, 2, 2, 0, 0), 1920, 1080, 'robidoux')
endcard = vrdf.KNLMCL(endcard, 2, 3, depth=16)
endcard = hvf.HQDeringmod(endcard, mrad=1, msmooth=1, mthr=80, minp=1, drrep=6, thr=24, planes=[0, 1, 2], show=False)
endcard = lvf.upscaled_sraa(endcard)
endcard = core.neo_f3kdb.Deband(endcard, 17, 36, 36, 36, 24, 0, sample_mode=4)
endcard = fvf.rfs(credit, endcard, mappings="[34791 35004]")

final = fvf.Depth(endcard, 10)

final.set_output()