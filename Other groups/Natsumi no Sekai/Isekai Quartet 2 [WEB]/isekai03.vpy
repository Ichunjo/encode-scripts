import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
import havsfunc as hvf
from cooldegrain import CoolDegrain
from mvsfunc import GetMatrix

core = vs.core
core.max_cache_size = 32000

src_a = core.ffms2.Source(r"[HorribleSubs] Isekai Quartet S2 - 03 [1080p].mkv")
endcard = core.imwri.Read(r"endcard03.jfif")
#comp = lvf.scomp(src, src_b, height=540)

src = fvf.Depth(src_a, 16)

den = CoolDegrain(src_a, tr=1, thsad=60, thsadc=48, bits=16, blksize=8, overlap=4)

aa = taa.TAAmbk(den, aatype='Eedi3', gamma=500, nrad=2, mdis=20, mtype=0)

adapt_m = vrdf.adaptive_mask(aa, 4).std.Invert()
db_a = dbs.f3kbilateral(aa, 17, 64, 64)
db_b = dbs.f3kbilateral(aa, 20, 84, 84)
db_c = dbs.f3kbilateral(aa, 22, 96, 96)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)

db = fvf.rfs(db, db_c, mappings='[14959 17039]')
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')

grain = core.grain.Add(db, 0.3, constant=True)

# Endcard
endcard = core.std.AssumeFPS(endcard, None, 24000, 1001)
endcard = core.resize.Spline36(endcard, grain.width, grain.height, grain.format, matrix_s=GetMatrix(src))
endcard = vrdf.KNLMCL(endcard, 0.75, 0.75, depth=16)
endcard = hvf.HQDeringmod(endcard, mrad=1, msmooth=1, mthr=80, minp=1, drrep=13, thr=24, planes=[0, 1, 2], show=False)
endcard = lvf.upscaled_sraa(endcard, 2, sharp_downscale=False)
endcard = core.neo_f3kdb.Deband(endcard, 18, 36, 36, 36, 32, 0)
endcard = endcard*223

merged = grain + endcard

final = fvf.Depth(merged, 10)

final.set_output()