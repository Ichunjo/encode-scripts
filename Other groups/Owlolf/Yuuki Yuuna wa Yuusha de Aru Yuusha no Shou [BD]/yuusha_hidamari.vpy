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

src = lvf.src(r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00003.m2ts')

#asrc = r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC1/BDMV/STREAM/00003.mka'
#ac.eztrim(src,[(0,-24)], asrc,"yuusha_hidamari.mka")

src = src[:-24]

opstart = 2614

#vrdf.generate_keyframes(src, 'yuusha_hidamari_keyframes.log')

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

desc = dsc.Descale(src32, 1280, 720, 'bicubic', 0.3782, 0.3109, gray=True)
desc_wasuyu = dsc.Descale(src32, 1280, 720, 'bicubic', 0, 1/2, gray=True)
desc = fvf.rfs(desc, desc_wasuyu, mappings=f'[28441 28791]')

den = CoolDegrain(desc, tr=1, thsad=60, thsadc=0, bits=16, blksize=8, overlap=4)
den_c = core.knlm.KNLMeansCL(src32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

db_y_a, db_c_a = vrdf.F3kdbSep(den, den_c, 15, 36, 36, 24, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(den, den_c, 15, 48, 48, 24, 0)
db_y = fvf.rfs(db_y_a, db_y_b, mappings=f'[{opstart} {opstart+49}] [{opstart+1886} {opstart+1911}] [5310 5536]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings=f'[{opstart} {opstart+49}] [{opstart+1886} {opstart+1911}] [5310 5536]')

scaled_y = znedi3_rpow2(db_y, 2, nns=4)
scaled_y = core.fmtc.resample(scaled_y, 1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32, csp=vs.GRAY16)
scaled = core.std.ShufflePlanes([scaled_y, db_c], [0, 1, 2], vs.YUV)

grain = kgf.adaptive_grain(scaled, 0.1)

credit_m = vrdf.DRM(src16, 720, 'bicubic', 0.3782, 0.3109).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, src16, credit_m), mappings='[2614 4770] [16640 16735] [32861 34812]')

final = fvf.Depth(credit, 10)

final.set_output()