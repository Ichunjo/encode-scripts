import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
import acsuite
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = lvf.src('[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00005.m2ts')
asrc = r"[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00005.mka"
ac.eztrim(src, [(24,-24)], asrc, "magia_bd_ncop.mka")

src = src[24:-24]
src = fvf.Depth(src, 16)

den_y = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=0)
den = core.knlm.KNLMeansCL(den_y, a=2, h=0.25, d=3, device_type='gpu', channels='UV')

aa = lvf.nneedi3_clamp(den, ret_mask=True, nns=4)

ref = fvf.Depth(aa, 8)
line_m_a = TAAmbk(ref, mtype=1, showmask=1).std.Binarize(10).rgvs.RemoveGrain(3).std.Inflate()
line_m_b = mdf.retinex_edgemask_mod(ref).std.Binarize(42).rgvs.RemoveGrain(3).rgvs.RemoveGrain(4).std.Inflate()
line_m = core.std.Expr([line_m_a, line_m_b], 'x y +')
line_m = core.std.Expr(line_m, 'x 30 < 0 x 3 * ?').std.Inflate()

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 18, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, fvf.Depth(line_m, 16))

grain = kgf.adaptive_grain(db, 0.25, luma_scaling=10)

final = fvf.Depth(grain, 10)

final.set_output()