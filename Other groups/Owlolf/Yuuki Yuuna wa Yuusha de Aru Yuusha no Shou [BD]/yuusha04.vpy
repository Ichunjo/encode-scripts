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

src = lvf.src(r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC2/BDMV/STREAM/00000.m2ts')

asrc = r'[BDMV][180530][Yuuki Yuuna wa Yuusha de Aru Yuusha no Shou][BD-BOX]/YUYUYU_YUUSYA_DISC2/BDMV/STREAM/00000.mka'
ac.eztrim(src,[(0,-24)], asrc,"yuusha04.mka")

src = src[:-24]

opstart = 2158

vrdf.generate_keyframes(src, 'yuusha04_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fe16 = fvf.Depth(fe, 16)
fe32 = fvf.Depth(fe, 32)

desc = dsc.Descale(fe32, 1280, 720, 'bicubic', 0, 1/2, gray=True)
desc_yuyuyu = dsc.Descale(fe32, 1280, 720, 'bicubic', 0.3782, 0.3109, gray=True)
desc = fvf.rfs(desc, desc_yuyuyu, mappings=f'[22556 23329]')

den = CoolDegrain(desc, tr=1, thsad=60, thsadc=0, bits=16, blksize=8, overlap=4)
den_c = core.knlm.KNLMeansCL(fe32, a=2, h=0.4, d=3, device_type='gpu', channels='UV')

aa_y = lvf.upscaled_sraa(den, rep=13, sharp_downscale=False)
aa_y = fvf.rfs(den, aa_y, mappings=f'[{opstart+1548} {opstart+1609}]')

db_y_a, db_c_a = vrdf.F3kdbSep(aa_y, den_c, 15, 30, 30, 20, 0)
db_y_b, db_c_b = vrdf.F3kdbSep(aa_y, den_c, 16, 54, 48, 24, 0)
db_y = fvf.rfs(db_y_a, db_y_b, mappings=f'[{opstart+767} {opstart+822}] [{opstart+1196} {opstart+1203}]')
db_c = fvf.rfs(db_c_a, db_c_b, mappings=f'[{opstart+767} {opstart+822}] [{opstart+1196} {opstart+1203}]')

scaled_y = znedi3_rpow2(db_y, 2, nns=4)
scaled_y = core.fmtc.resample(scaled_y, 1920, 1080, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32, csp=vs.GRAY16)
scaled = core.std.ShufflePlanes([scaled_y, db_c], [0, 1, 2], vs.YUV)

grain = kgf.adaptive_grain(scaled, 0.1)

credit_m = vrdf.DRM(fe16, 720, 'bicubic', 0, 1/2).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, fe16, credit_m), mappings='[2158 4316] [13919 14234] [32056 34213]')

final = fvf.Depth(credit, 10)

final.set_output()