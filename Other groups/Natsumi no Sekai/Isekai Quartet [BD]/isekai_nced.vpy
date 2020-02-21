import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vsTAAmbk as taa
import debandshit as dbs
from cooldegrain import CoolDegrain
import acsuite

ac = acsuite.AC()

core = vs.core
core.max_cache_size = 16384


src = lvf.src("IsekaiQuartet_Season1/BDMV/STREAM/00030.m2ts")


asrc = r'IsekaiQuartet_Season1/BDMV/STREAM/nced.w64'
ac.eztrim(src,[(24,-24)], asrc,"nced-cut.mka")

src = src[24:-24]


shift = core.fmtc.resample(src, sx=[0, 0.5], kernel='spline36')
den = CoolDegrain(shift, tr=1, thsad=80, thsadc=60, bits=16, blksize=8, overlap=4)
aa = taa.TAAmbk(den, aatype='Nnedi3', mtype=0)
db = dbs.f3kbilateral(aa, 17, 76, 76)
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')
grain = core.grain.Add(db, 0.3, constant=True)

final = fvf.Depth(grain, 10)

final.set_output()