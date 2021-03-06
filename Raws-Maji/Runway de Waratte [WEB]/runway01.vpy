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

core = vs.core
core.max_cache_size = 26000

src_a = lvf.src(r'WKN/Smile_Down_01_FR_HD.mp4')
src_b = lvf.src(r'AMZ/Runway de Waratte - 01 (Amazon dAnime VBR 1080p).mkv')

src_a = src_a[0] + src_a
# scomp = lvf.stack_compare(src_a, src_b, height=540)

vrdf.generate_keyframes(src_b, 'runway01_keyframes.log')

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade_a = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade_a = core.std.MaskedMerge(src_a, src_b, src_fade_a)
src = fvf.rfs(src, src_fade_a,
    mappings='[700 792] [868 944] [2635 2670] [3455 3543] [5341 5402] [6396 6470] [7100 7133] [8398 8483] [8253 8290] [9224 9281]' +
     '[13082 13144] [16993 17098] [30977 31034] [31285 31335] [33898 34081] [34132 34218] [34831 34932] [35088 35244]'
)
# scomp2 = lvf.stack_compare(src, src_b, height=540)

desc_y = dsc.Descale(src, 1600, 900, 'bicubic', 0, 1 / 2, gray=True)
down_y = core.resize.Spline36(get_y(src), 1600, 900)

credit_m = vrdf.DRM(src, 900, 'bicubic', 0, 1 / 2).std.Inflate()
credit_m = core.resize.Spline36(credit_m, 1600, 900)
desc_y = fvf.rfs(
    desc_y, core.std.MaskedMerge(desc_y, down_y, credit_m), mappings='[945 2523] [3455 3543] [32474 34724]')

planes = split(src)
planes[1], planes[2] = [
    core.resize.Spline36(
        plane, desc_y.width / 2, desc_y.height / 2, src_left=vrdf.GetChromaShift(src.height, desc_y.height))
    for plane in planes[1:]
]

scaled = join([desc_y, planes[1], planes[2]])

den = CoolDegrain(scaled, tr=1, thsad=48, thsadc=48, bits=16, blksize=8, overlap=4)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, mtype=0, down8=False, opencl=True)
planes = kgf.split(den)
planes = [lvf.upscaled_sraa(plane, 1.5, None, None, False) for plane in planes[0:]]
aa_b = kgf.join(planes)
aa_c = taa.TAAmbk(den, aatype='Eedi3SangNom', cycle=2, mtype=0, down8=False)
aa = fvf.rfs(aa_a, aa_b, mappings='[6570 6663]')
aa = fvf.rfs(aa_a, aa_c, mappings='[18177 18254]')

db_m = vrdf.adaptive_mask(aa, 20)
db_a = core.neo_f3kdb.Deband(aa, 17, 48, 48, 48, 32, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 18, 56, 56, 56, 32, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 20, 64, 64)
db = core.std.MaskedMerge(db_a, db_b, db_m)
db = fvf.rfs(db, db_c, mappings='[286 342] [639 693]')

grain = kgf.adaptive_grain(db, 0.2)

final = fvf.Depth(grain, 10)

final.set_output()