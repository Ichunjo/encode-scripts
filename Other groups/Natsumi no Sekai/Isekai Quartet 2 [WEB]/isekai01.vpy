import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000

src = lvf.src(r"Isekai Quartet2 - 01 [1080p].mkv")
#src_b = lvf.src(r"Isekai Quartet S2 - 01 (Amazon Prime VBR 720p).mkv")
#comp = lvf.scomp(src, src_b, height=540)

vrdf.generate_keyframes(src, 'isekai01_keyframes.log')

src16 = fvf.Depth(src, 16)

den = CoolDegrain(src16, tr=1, thsad=60, thsadc=48, bits=16, blksize=8, overlap=4)

aa = taa.TAAmbk(den, aatype='Eedi3', gamma=500, nrad=2, mdis=20, mtype=2)

adapt_m = vrdf.adaptive_mask(aa, 4).std.Invert()
db_a = dbs.f3kbilateral(aa, 17, 64, 64)
db_b = dbs.f3kbilateral(aa, 20, 84, 84)
db_c = dbs.f3kbilateral(aa, 22, 96, 96)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)

db = fvf.rfs(db, db_c, mappings='[3722 3833]')
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')

grain = core.grain.Add(db, 0.3, constant=True)

final = fvf.Depth(grain, 10)

final.set_output()