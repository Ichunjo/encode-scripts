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

src_nced = 	core.dgdecodenv.DGSource(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00005.dgi")

#asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00005.mka'
#ac.eztrim(src_nced,[(24,-24)], asrc,"oresuki_nced.mka")

src = 	src_nced[24:-24]

#vrdf.generate_keyframes(src, 'oresuki_nced_keyframes.log')

#########
desc_str = 0.5

fe = core.edgefixer.ContinuityFixer(src, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])

y = kgf.get_y(fvf.Depth(fe, 32))

desc_sharp =    dsc.Descale(y, 1280, 720, 'bilinear')
desc_regular =  dsc.Descale(y, 1280, 720, 'bicubic', 0, 1/2)
desc_y =        core.std.Expr([desc_sharp, desc_regular], f'x {desc_str} * y 1 {desc_str} - * +')

den_y =     CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_uv =    CoolDegrain(fe, tr=1, thsad=0, thsadc=60, bits=16, blksize=8, overlap=4)

line_m = kgf.retinex_edgemask(den_y).std.Binarize(9500)
line_m = line_m.rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate()
darken = hvf.FastLineDarkenMOD(den_y, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0)
aa = TAAmbk(darken, aatype='Nnedi3', mtype=0, down8=False)
aa = core.std.MaskedMerge(den_y, aa, line_m)

full = mdf.to_444(den_uv, 1280, 720, False)
scaled = kgf.join([den_y, full[0], full[1]])

db = core.neo_f3kdb.Deband(scaled, 16, 30, 30, 30, 12, 0, sample_mode=4)
grain = kgf.adaptive_grain(db)

final = fvf.Depth(grain, 10)

final.set_output()