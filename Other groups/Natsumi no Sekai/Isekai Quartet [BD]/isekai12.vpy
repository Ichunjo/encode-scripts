import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
from cooldegrain import CoolDegrain
from mvsfunc import GetMatrix
import acsuite

ac = acsuite.AC()

core = vs.core
core.max_cache_size = 16384

src = lvf.src("IsekaiQuartet_Season1/BDMV/STREAM/00058.m2ts")
endcard = core.imwri.Read('yande.re 548179 ainz_ooal_gown crossover isekai_quartet natsuki_subaru overlord satou_kazuma tagme tanya_degurechaff uniform youjo_senki.jpg', float_output=True)
endcard = core.std.AssumeFPS(endcard, src)

asrc = r'IsekaiQuartet_Season1/BDMV/STREAM/12.w64'
ac.eztrim(src,[(24,-24)], asrc,"isekai12-cut.mka")

src = src[24:-24]


shift = core.fmtc.resample(src, sx=[0, 0.5], kernel='spline36')
den = CoolDegrain(shift, tr=1, thsad=80, thsadc=60, bits=16, blksize=8, overlap=4)
aa = taa.TAAmbk(den, aatype='Nnedi3', mtype=0)
db = dbs.f3kbilateral(aa, 17, 76, 76)
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')
grain = core.grain.Add(db, 0.3, constant=True)



# Endcard
endcard = core.resize.Spline36(endcard, grain.width, grain.height, grain.format, matrix_s=GetMatrix(src))
endcard = core.neo_f3kdb.Deband(endcard, 18, 36, 36, 36, 32, 0)
endcard = endcard*239

merged = grain + endcard



final = fvf.Depth(merged, 10)

final.set_output()
