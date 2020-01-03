import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00001.m2ts")

asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00001.mka'
ac.eztrim(src,[(0,-25)], asrc,"oresuki01.mka")

src = src[0:-25]

vrdf.generate_keyframes(src, 'oresuki01_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])

fe16 = fvf.Depth(fe, 16)

desc_y = dsc.Descale(fe16, 1280, 720, 'bicubic', 0, 1/2, gray=True)

den_y = CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_uv = CoolDegrain(fe16, tr=1, thsad=0, thsadc=48, bits=16, blksize=8, overlap=4)

scaled_y = nnedi3_rpow2CL(den_y, 2, nns=4).resize.Spline36(1920, 1080)

scaled = core.std.ShufflePlanes([scaled_y, den_uv], [0, 1, 2], vs.YUV)

aa = lvf.upscaled_sraa(fe16, rep=13, sharp_downscale=False)
scaled = fvf.rfs(scaled, aa, mappings='[726 815]')

darken = hvf.FastLineDarkenMOD(scaled, strength=48, protection=7, luma_cap=191, threshold=4, thinning=0)

line_m = kgf.retinex_edgemask(scaled)
db = core.neo_f3kdb.Deband(scaled, 16, 30, 30, 30, 12, 0, sample_mode=4)
db = core.std.MaskedMerge(db, darken, line_m)

grain = kgf.adaptive_grain(db, 0.2)

credit_m = vrdf.DRM(fe16, 720, 'bicubic', 0, 1/2).std.Inflate()

credit = fvf.rfs(grain, core.std.MaskedMerge(grain, fe16, credit_m, 0), mappings='[0 725]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings='[833 933]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m[20670], 0), mappings='[20630 20677]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings='[20678 20737]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings='[20866 20951]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings='[31737 34046]')

aa = lvf.upscaled_sraa(fe16, 2, rep=13, sharp_downscale=True)

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[1124], 0), mappings='[1060 1124]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[1125 1229]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[1768], 0), mappings='[1716 1768]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[1769 1887]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[2782], 0), mappings='[2665 2830]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[4861], 0), mappings='[4813 4865]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[4866 4947]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[8202], 0), mappings='[8179 8272]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[13810], 0), mappings='[13757 13810]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[21409], 0), mappings='[21338 21409]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[21480 21520]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[24786], 0), mappings='[24719 24786]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[24829 24869]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[28298], 0), mappings='[28206 28298]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0), mappings='[28299 28426]')

final = fvf.Depth(credit, 10)
final.set_output()