import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import modfunc as mdf
import debandshit as dbs
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrainSF
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 26000

ac = acsuite.AC()

src = 		core.dgdecodenv.DGSource(r'DMPXA048B/BDMV/STREAM/00006.dgi')
src_ncop = 	core.dgdecodenv.DGSource(r'DMPXA047B/BDMV/STREAM/00008.dgi')
src_nced = 	core.dgdecodenv.DGSource(r'DMPXA047B/BDMV/STREAM/00009.dgi')

opstart, opend = 1104, 3260
edstart, edend = 32296, 34453
opening_a = f'[{opstart+1576} {opstart+1583}] [{opstart+998} {opstart+1026}]'
opening_b = f'[{opstart} {opstart+408}]'

detele = core.eedi3m.EEDI3CL(src, 1)
detele = fvf.rfs(src, detele, mappings=opening_a)

fe = core.edgefixer.ContinuityFixer(detele, 2, 2, 2, 2)
fe32 = fvf.Depth(fe, 32)

desc_y = dsc.Descale(fe32, 1280, 720, 'bicubic', 0, 1/2, gray=True)
den_y = CoolDegrainSF(desc_y, tr=2, thsad=80, blksize=8, overlap=4, plane=0)
scaled_y_a = nnedi3_rpow2CL(den_y, 2, nns=4).fmtc.resample(1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32)
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

# Black frames before ending
ref = fvf.Depth(fe, 16)
credit = fvf.rfs(grain, ref, mappings=f'[32236 {edstart-1}]')

# Episodes names
aa_cred = TAAmbk(ref, aatype='Nnedi3', nsize=0, nns=4, mtype=1, opencl=True)
credit = fvf.rfs(credit, aa_cred, mappings='[34454 34519] [34953 35024]')

# Endcard (?)
credit_m = vrdf.DRM(ref, 720, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(credit, core.std.MaskedMerge(grain, ref, credit_m), mappings='[35174 {src.num_frames-1}]')

# Opening / Ending credit
credit_op_m = vrdf.DCM(credit, src[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 2, 2)
credit_ed_m = vrdf.DCM(credit, src[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 2, 2)
credit_m = core.std.Expr([credit_op_m, credit_ed_m], 'x y +')
credit = core.std.MaskedMerge(credit, ref, credit_m)

final = fvf.Depth(credit, 10)

final.set_output()