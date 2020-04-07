import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
from cooldegrain import CoolDegrain
from mvsfunc import GetMatrix
core = vs.core
core.max_cache_size = 16384

src = lvf.src(r"[HorribleSubs] Isekai Quartet S2 - 10 [1080p].mkv")
#src_b = lvf.src(r"Isekai Quartet S2 - 06 (Amazon Prime VBR 720p).mkv")
endcard = core.ffms2.Source(r"endcard10.jpg")
#comp = lvf.scomp(src_a, src_b, height=540)

src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, thsadc=48, bits=16, blksize=8, overlap=4)

aa = taa.TAAmbk(den, aatype='Nnedi3', mtype=0)

adapt_m = vrdf.adaptive_mask(aa, 4).std.Invert()
db_a = dbs.f3kbilateral(aa, 17, 64, 64)
db_b = dbs.f3kbilateral(aa, 20, 84, 84)
db_c = dbs.f3kbilateral(aa, 22, 96, 96)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)

db = fvf.rfs(db, db_c, mappings='[13572 15642]')
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')

grain = core.grain.Add(db, 0.3, constant=True)

# Endcard
endcard = core.std.AssumeFPS(endcard, grain)
endcard = core.resize.Spline36(endcard, grain.width, grain.height, grain.format, matrix_s=GetMatrix(src))
endcard = lvf.upscaled_sraa(endcard, 2, sharp_downscale=False)
endcard = core.neo_f3kdb.Deband(endcard, 18, 36, 36, 36, 32, 0)
endcard = endcard*223

merged = grain + endcard

final = fvf.Depth(merged, 10)

final.set_output()