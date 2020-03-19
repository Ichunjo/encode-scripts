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
src = 		lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00007.m2ts')
src_prev = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00010.m2ts')
src_ncop = 	lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00009.m2ts')
encard = 	lvf.src(r'episode-3-the-fool-and-death-733627 [00_23_42.087] 1.png')
encard = core.std.AssumeFPS(encard, fpsnum=src.fps.numerator, fpsden=src.fps.denominator)
encard = core.resize.Spline36(encard, format=src.format, matrix_s=mvf.GetMatrix(src, dIsRGB=True))

# Audio cuts
asrc = 		r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00007.mka'
asrc_prev = r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 2/BDMV/STREAM/00010.mka'
ablank = 	r'blank.wav'
ac.eztrim(src,	[(24,34070)], asrc, 	'rokuaka03a.mka')
ac.eztrim(src,	[(24,768)], asrc_prev, 	'rokuaka03b.mka')
ac.eztrim(src,	[(0,215)], 	ablank, 	'rokuaka03c.mka')

# Video cuts
src = src[24:34070]
src_prev = src_prev[24:768]
src_ncop = src_ncop[24:-24]

# 0:24:20.000
src = src+src_prev+(encard*215)

# Variables
opstart, opend = 31888, 34045

# Keyframes for --qpfile
vrdf.generate_keyframes(src, 'rokuaka03_keyframes.log')

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
aa_a = taa.TAAmbk(merge, aatype='Nnedi3', nsize=4, nns=4, mclip=line_m, opencl=True)
aa_b = taa.TAAmbk(merge, aatype='Eedi3SangNom', mtype=2, cycle=1, opencl=True)
aa = fvf.rfs(aa_a, aa_b, mappings='[26161 26304]')

# Deband
db_a = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, aa, line_m)

# Random 1080p things
credit_m = vrdf.DRM(src16, 837, 'bicubic', 0.2620, 0.3690, 80).std.Inflate()
credit = fvf.rfs(db, core.std.MaskedMerge(db, src16, credit_m[118]), mappings='[744 827]')

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30207], left=0, right=958, top=0, bottom=0)), mappings='[30187 30274]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30351], left=0, right=0, top=0, bottom=144)), mappings='[30308 30397]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30450], left=0, right=697, top=0, bottom=138)), mappings='[30410 30498]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30559], left=0, right=953, top=0, bottom=0)), mappings='[30530 30643]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30744], left=681, right=0, top=483, bottom=0)), mappings='[30686 30793]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30857], left=0, right=1049, top=0, bottom=652)), mappings='[30831 30926]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[30991], left=0, right=0, top=0, bottom=676)), mappings='[30939 31034]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m[31057]), mappings='[31049 31124]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[31202], left=0, right=16, top=591, bottom=0)), mappings='[31157 31243]')

credit_m_a = vrdf.RegionMask(credit_m[31294], left=0, right=1302, top=0, bottom=34)
credit_m_b = vrdf.RegionMask(credit_m[31294], left=1535, right=0, top=0, bottom=0)
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, core.std.Expr([credit_m_a, credit_m_b], 'x y +')), mappings='[31268 31375]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, vrdf.RegionMask(credit_m[31532], left=0, right=0, top=668, bottom=0)), mappings='[31504 31599]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, src16, credit_m), mappings='[31708 31887]')

# Opening credit
credit_op_m = vrdf.DCM(credit, src[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 4, 4)
credit_op = core.std.MaskedMerge(credit, src16, credit_op_m)

# Galaxy brain filtering for the endcard
endcard = vrdf.Bicubic(core.std.Crop(src16, 2, 2, 0, 0), 1920, 1080, 'robidoux')
endcard = vrdf.KNLMCL(endcard, 1, 1, depth=16)
endcard = hvf.HQDeringmod(endcard, mrad=1, msmooth=1, mthr=80, minp=1, drrep=13, thr=24, planes=[0, 1, 2], show=False)
endcard = core.neo_f3kdb.Deband(endcard, 17, 36, 36, 36, 24, 0, sample_mode=4)
endcard = fvf.rfs(credit_op, endcard, mappings="[34790 35004]")

final = fvf.Depth(endcard, 10)

final.set_output()