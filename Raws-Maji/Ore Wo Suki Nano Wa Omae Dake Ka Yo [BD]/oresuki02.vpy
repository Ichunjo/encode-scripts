import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = 		lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00002.m2ts")
src_ncop = 	lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00004.m2ts")
src_nced = 	lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00005.m2ts")

asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00002.mka'
ac.eztrim(src,[(0,34045)], asrc,"oresuki02.mka")

src = 		src[0:34045]
src = 		src+src[-1]
src_ncop = 	src_ncop[24:-24]
src_nced = 	src_nced[24:-24]

opstart, opend = 456, 2613
edstart, edend = 31767, 33924

vrdf.generate_keyframes(src, 'oresuki02_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])

fe16 = fvf.Depth(fe, 16)

desc_y = dsc.Descale(fe16, 1280, 720, 'bicubic', 0, 1/2, gray=True)

den_y = CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_uv = CoolDegrain(fe16, tr=1, thsad=0, thsadc=48, bits=16, blksize=8, overlap=4)

scaled_y = nnedi3_rpow2CL(den_y, 2, nns=4).resize.Spline36(1920, 1080)

scaled = core.std.ShufflePlanes([scaled_y, den_uv], [0, 1, 2], vs.YUV)

darken = hvf.FastLineDarkenMOD(scaled, strength=48, protection=7, luma_cap=191, threshold=4, thinning=0)

line_m = kgf.retinex_edgemask(scaled)
db = core.neo_f3kdb.Deband(scaled, 16, 30, 30, 30, 12, 0, sample_mode=4)
db = core.std.MaskedMerge(db, darken, line_m)

grain = kgf.adaptive_grain(db, 0.2)

credit_op_m = vrdf.DCM(grain, src[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 4, 4).std.Deflate()
credit_op = core.std.MaskedMerge(grain, fe16, credit_op_m)
credit_ed_m = vrdf.DCM(grain, src[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 4, 4).std.Deflate()
credit_ed = core.std.MaskedMerge(credit_op, fe16, credit_ed_m)

credit_m = vrdf.DRM(fe16, 720, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(credit_ed, core.std.MaskedMerge(credit_ed, fe16, credit_m, 0), mappings='[2660 2733]')
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, fe16, credit_m, 0), mappings='[33925 34045]')

final = fvf.Depth(credit, 10)
final.set_output()