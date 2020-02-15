import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import debandshit as dbs
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrainSF
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 26000

ac = acsuite.AC()

src = 		core.dgdecodenv.DGSource(r'DMPXA048B/BDMV/STREAM/00007.dgi')
src_nced = 	core.dgdecodenv.DGSource(r'DMPXA047B/BDMV/STREAM/00009.dgi')

edstart, edend = 32296, 34453

fe = core.edgefixer.ContinuityFixer(src, 2, 2, 2, 2)
fe32 = fvf.Depth(fe, 32)

desc_y = dsc.Descale(fe32, 1280, 720, 'bicubic', 0, 1/2, gray=True)
den_y = CoolDegrainSF(desc_y, tr=2, thsad=80, blksize=8, overlap=4, plane=0)
scaled_y = nnedi3_rpow2CL(den_y, 2, nns=4).fmtc.resample(1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32)

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

# Black frames before ending
ref = fvf.Depth(fe, 16)

# Endcard (?)
credit_m = vrdf.DRM(ref, 720, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, ref, credit_m), mappings='[28739 31656] [34454 34525]')

# Ending credit
credit_ed_m = vrdf.DCM(credit, src[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 2, 2)
credit = core.std.MaskedMerge(credit, ref, credit_ed_m)

final = fvf.Depth(credit, 10)

final.set_output()