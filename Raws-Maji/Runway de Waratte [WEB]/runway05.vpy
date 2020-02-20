import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import descale as dsc
import debandshit as dbs
from vsutil import get_y, join, split
from cooldegrain import CoolDegrain
from edi_rpow2 import znedi3_rpow2

core = vs.core
core.max_cache_size = 26000

src_a = core.dgdecodenv.DGSource(r'WKN/Smile_Down_05_FR_HD.dgi')
src_b = core.dgdecodenv.DGSource(r'AMZ/Runway de Waratte - 05 (Amazon dAnime VBR 1080p).dgi')

src_a = src_a[0] + src_a + src_a[-1] *2
scomp = lvf.stack_compare(src_a, src_b, height=540)

#vrdf.generate_keyframes(src_b, 'runway05_keyframes.log')

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade_a = kgf.hardsubmask_fades(src_a, src_b, highpass=1300)
src_fade_a = core.std.MaskedMerge(src_a, src_b, src_fade_a)
src = fvf.rfs(src, src_fade_a, mappings='[5781 5867] [4117 4203] [17956 18001] [23590 23673] [34931 35244]')
#scomp2 = lvf.stack_compare(src, src_b, height=540)

opstart, opend = 3597, 5753
edstart, edend = 32776, 34932

desc_y = dsc.Descale(src, 1600, 900, 'bicubic', 0, 1 / 2, gray=True)
desc_y_ed = dsc.Descale(desc_y, 1280, 720, 'bicubic', 0, 1 / 2)
desc_y_ed = znedi3_rpow2(desc_y_ed, 2, nns=4).resize.Spline36(1600, 900)
desc_y = fvf.rfs(desc_y, desc_y_ed, mappings=f'[{edstart} {edend}]')

down_y = core.resize.Spline36(get_y(src), 1600, 900)

credit_m = vrdf.DRM(src, 900, 'bicubic', 0, 1 / 2).std.Inflate()
credit_m = core.resize.Spline36(credit_m, 1600, 900)
desc_y = fvf.rfs(
    desc_y, core.std.MaskedMerge(desc_y, down_y, credit_m), mappings=f'[{opstart} {opend}] [{edstart} {edend}]')

planes = split(src)
planes[1], planes[2] = [
    core.resize.Spline36(
        plane, desc_y.width / 2, desc_y.height / 2, src_left=vrdf.GetChromaShift(src.height, desc_y.height))
    for plane in planes[1:]
]

scaled = join([desc_y, planes[1], planes[2]])

den = CoolDegrain(scaled, tr=1, thsad=48, thsadc=48, bits=16, blksize=8, overlap=4)

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, mtype=0, down8=False, opencl=True)

db_m = vrdf.adaptive_mask(aa, 16)
db_a = core.neo_f3kdb.Deband(aa, 17, 48, 48, 48, 32, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 18, 56, 56, 56, 32, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 20, 48, 42)
db = core.std.MaskedMerge(db_a, db_b, db_m)
db = fvf.rfs(db, db_c, mappings='[16401 16510]')

grain = kgf.adaptive_grain(db, 0.2)

final = fvf.Depth(grain, 10)

final.set_output()