import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import descale as dsc
import acsuite
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = core.dgdecodenv.DGSource(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00001.dgi")

asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00001.mka'
ac.eztrim(src,[(0,-25)], asrc,"oresuki01.mka")

src = src[:-25]

desc_str = 0.5
desc_w, desc_h = 1280, 720
vrdf.generate_keyframes(src, 'oresuki01_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [2,1,1], [2,1,1], [2,1,1], [2,1,1])

y = kgf.get_y(fvf.Depth(fe, 32))

desc_sharp =    dsc.Descale(y, desc_w, desc_h, 'bilinear')
desc_regular =  dsc.Descale(y, desc_w, desc_h, 'bicubic', 0, 1/2)
desc_y =        core.std.Expr([desc_sharp, desc_regular], f'x {desc_str} * y 1 {desc_str} - * +')

den_y = CoolDegrain(desc_y, tr=1, thsad=48, bits=16, blksize=8, overlap=4, plane=0)
den_uv = CoolDegrain(fe, tr=1, thsad=0, thsadc=60, bits=16, blksize=8, overlap=4, plane=3)

line_m = mdf.retinex_edgemask_mod(den_y, opencl=True).std.Binarize(9500)
line_m = line_m.rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate()
darken = hvf.FastLineDarkenMOD(den_y, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0)
aa = TAAmbk(darken, aatype='Nnedi3', mtype=0, down8=False)
aa = core.std.MaskedMerge(den_y, aa, line_m)

u, v = vrdf.to444(den_uv, 1280, 720, False)
scaled = kgf.join([den_y, u, v])

db = core.neo_f3kdb.Deband(scaled, 16, 30, 30, 30, 12, 0, sample_mode=4)
grain = kgf.adaptive_grain(db)

# 1080p things
fe16 = fvf.Depth(fe, 16)
credit_m = core.resize.Spline36(vrdf.DRM(fe16, desc_h, 'bicubic', 0, 1/2).std.Inflate(), desc_w, desc_h)

ref = core.resize.Spline36(kgf.get_y(fe16), desc_w, desc_h)
grain_y = kgf.get_y(grain)

credit = fvf.rfs(grain_y, core.std.MaskedMerge(grain_y, ref, credit_m, 0),      mappings='[0 725]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m, 0),        mappings='[833 933]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m[20670], 0), mappings='[20630 20677]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m, 0),        mappings='[20678 20737]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m, 0),        mappings='[20866 20951]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m, 0),        mappings='[31737 34046]')

aa = lvf.upscaled_sraa(kgf.get_y(fe16), 2, rep=13, h=desc_h, sharp_downscale=True)

credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[1124], 0),   mappings='[1060 1124]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[1125 1229]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[1768], 0),   mappings='[1716 1768]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[1769 1887]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[2782], 0),   mappings='[2665 2830]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[4861], 0),   mappings='[4813 4865]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[4866 4947]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[8202], 0),   mappings='[8179 8272]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[13810], 0),  mappings='[13757 13810]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[21409], 0),  mappings='[21338 21409]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[21480 21520]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[24786], 0),  mappings='[24719 24786]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[24829 24869]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m[28298], 0),  mappings='[28206 28298]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, aa, credit_m, 0),         mappings='[28299 28426]')

# Merging the chroma
credit = core.std.ShufflePlanes([credit, grain], [0, 1, 2], vs.YUV)

final = fvf.Depth(credit, 10)

final.set_output()