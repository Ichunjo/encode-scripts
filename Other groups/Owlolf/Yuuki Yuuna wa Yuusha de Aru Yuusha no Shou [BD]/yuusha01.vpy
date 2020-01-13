import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import descale as dsc
import acsuite
from edi_rpow2 import znedi3_rpow2
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = lvf.src(r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00000.m2ts')

asrc = r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00000.mka'
ac.eztrim(src,[(0,34813)], asrc,"yuusha01.mka")

src = src[:34813]
src = src+src[-1]

vrdf.generate_keyframes(src, 'yuusha01_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fe16 = fvf.Depth(fe, 16)
fe32 = fvf.Depth(fe, 32)

desc = dsc.Descale(fe32, 1280, 720, 'bicubic', 0, 1/2, gray=True)

den = CoolDegrain(desc, tr=1, thsad=60, thsadc=0, bits=16, blksize=8, overlap=4)
den_c = core.knlm.KNLMeansCL(fe32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

db_y, db_c = vrdf.F3kdbSep(den, den_c, 15, 30, 30, 20, 0)

scaled_y = znedi3_rpow2(db_y, 2, nns=4)
scaled_y = core.fmtc.resample(scaled_y, 1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32, csp=vs.GRAY16)
scaled = core.std.ShufflePlanes([scaled_y, db_c], [0, 1, 2], vs.YUV)

grain = kgf.adaptive_grain(scaled, 0.1)

credit_m = vrdf.DRM(fe16, 720, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, fe16, credit_m), mappings='[0 1774] [32296 34452]')

final = fvf.Depth(credit, 10)

final.set_output()