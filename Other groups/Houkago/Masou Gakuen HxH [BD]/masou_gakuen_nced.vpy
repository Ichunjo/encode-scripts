from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk
from acsuite import eztrim
import atomchtools as atf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 8192


path = "魔装学園H×H/MASOUGAKUEN_HH_2/BDMV/STREAM/00010."
src = lvf.src(path + "m2ts")
asrc = path + "wav"
eztrim(src, (24, -24), asrc, "masou_gakuen_nced.mka")

src = src[24:-24]

fe = lvf.edgefixer(src, [1, 1, 1])

den = CoolDegrain(fe, tr=1, thsad=48, bits=16, blksize=8, overlap=4, plane=4)

line_m = atf.retinex_edgemask(den, 0.5, draft=True, opencl=True)
line_m = lvf.iterate(line_m, core.std.Inflate, 4)
line_m = core.std.Expr(line_m, 'x 8000 < 0 x 2 * ?')

aa = TAAmbk(den, aatype='Eedi3', mclip=line_m)

db = core.placebo.Deband(aa, radius=17, threshold=5.5, iterations=1, grain=0, planes=1|2|4)
db = core.std.MaskedMerge(db, aa, line_m)

grain = core.placebo.Deband(db, radius=0, threshold=0, iterations=0, grain=6.75, planes=1)

final = core.fmtc.bitdepth(grain, bits=10, dmode=3)
final.set_output()
