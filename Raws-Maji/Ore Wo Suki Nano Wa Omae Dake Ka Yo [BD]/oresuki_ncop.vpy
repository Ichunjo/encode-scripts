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

src_ncop = 	lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00004.m2ts")

asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00004.mka'
ac.eztrim(src_ncop,[(24,-24)], asrc,"oresuki_ncop.mka")

src_ncop = 	src_ncop[24:-24]

vrdf.generate_keyframes(src_ncop, 'oresuki_ncop_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src_ncop, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])

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

grain = kgf.adaptive_grain(db, 0.1, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()