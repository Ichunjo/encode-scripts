import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import mvsfunc as mvf
import vardefunc as vrdf
import descale as dsc
import debandshit as dbs
import modfunc as mdf
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrainSF
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

ac = acsuite.AC()

src = 	core.dgdecodenv.DGSource(r'DMPXA047B/BDMV/STREAM/00008.dgi')

opstart = 0
opening_a = f'[{opstart+998} {opstart+1026}] [{opstart+1576} {opstart+1583}]'
opening_b = f'[{opstart} {opstart+408}]'

detele = core.eedi3m.EEDI3CL(src, 1)
detele = fvf.rfs(src, detele, mappings=opening_a)

fe = core.edgefixer.ContinuityFixer(detele, 2, 2, 2, 2)
fe32 = fvf.Depth(fe, 32)

desc_y = dsc.Descale(fe32, 1280, 720, 'bicubic', 0, 1/2, gray=True)
den_y = CoolDegrainSF(desc_y, tr=2, thsad=80, blksize=8, overlap=4, plane=0)
scaled_y_a = nnedi3_rpow2CL(den_y, 2, nns=4)
scaled_y_a = core.fmtc.resample(scaled_y_a, 1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32)
scaled_y_b = mdf.upscaled_sraa_mod(den_y, 2, 13, 1080)
scaled_y = fvf.rfs(scaled_y_a, scaled_y_b, mappings=opening_b)

den_uv = core.knlm.KNLMeansCL(fe32, a=2, h=0.5, d=3, device_type='gpu', channels='UV')

scaled = core.std.ShufflePlanes([scaled_y, den_uv], [0, 1, 2], vs.YUV)
scaled = fvf.Depth(scaled, 16)

adapt_m = vrdf.adaptive_mask(scaled, 25)
db_a = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(scaled, 17, 48, 42, 42, 24, 0, sample_mode=4)
db_b = core.std.MaskedMerge(db_a, db_b, adapt_m)
db_c = core.neo_f3kdb.Deband(dbs.f3kbilateral(scaled, 17, 64, 48), grainy=24, preset='depth')
db = lvf.limit_dark(db_a, db_b, threshold=0.25)
db = lvf.limit_dark(db, db_c, threshold=0.09)

grain = kgf.adaptive_grain(db, luma_scaling=6)

final = fvf.Depth(grain, 10)
final.set_output()