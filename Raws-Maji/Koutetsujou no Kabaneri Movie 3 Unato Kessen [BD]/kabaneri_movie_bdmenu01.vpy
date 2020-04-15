from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
import lvsfunc as lvf
import fvsfunc as fvf
import modfunc as mdf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

src = lvf.src('[191211][BDMV] 甲鉄城のカバネリ 海門決戦/KABANERI_UNATO/BDMV/STREAM/00002.m2ts')

src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)

aa = TAAmbk(den, aatype='Nnedi3', mtype=1, nsize=2)

line_m = mdf.retinex_edgemask_mod(aa).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(aa, 16, 36, 36, 36, 24, 0, sample_mode=4)

db = core.std.MaskedMerge(db, aa, line_m)

final = fvf.Depth(db, 10)

final.set_output()
