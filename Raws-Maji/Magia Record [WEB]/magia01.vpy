import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000
ac = acsuite.AC()

src_a = lvf.src(r'WKN/Magia_Record_01_FR_HD.mp4')
src_b = lvf.src(r'Funi/[HorribleSubs] Magia Record - 01 [1080p].mkv')[289:]

asrc = r"WKN/Magia_Record_01_FR_HD.mka"
ac.eztrim(src_a, [(0,1)], 								asrc, "Magia01a_cut.mka")
ac.eztrim(src_a, [(0,18150),(18154,33930),(33948,0)], 	asrc, "Magia01b_cut.mka")
ac.eztrim(src_a, [(0,12)], 								asrc, "Magia01c_cut.mka")

src_a = src_a[0]+src_a[:18150]+src_a[18154:33930]+src_a[33948:]+(src_a[-1]*12)
src_b = src_b[:-1]

vrdf.generate_keyframes(src_a, 'magia01_keyframes.log')

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade_a = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade_b = kgf.hardsubmask_fades(src_a, src_b, highpass=1300)
src_fade_a = core.std.MaskedMerge(src_a, src_b, src_fade_a)
src_fade_b = core.std.MaskedMerge(src_a, src_b, src_fade_b)

src = fvf.rfs(src, src_fade_a, mappings="[7127 7177] [7301 7366] [9252 9749] [9838 10137] [10525 10726] [13398 13451] [13518 13583] [15150 15491] [25512 25652] [26711 26824] [29453 29476] [29510 29532] [29640 29663] [29775 29798] [29866 29889] [30011 30034] [30361 30844] [31743 31768] [33741 33746] [33927 34045]")
src = fvf.rfs(src, src_fade_b, mappings="[32079 32281] [33861 33926]")

#scomp2 = lvf.stack_compare(src, src_b, height=540)

den = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=2, down8=False, opencl=True)

adapt_m = vrdf.adaptive_mask(aa, 28)
adapt_m_2 = vrdf.adaptive_mask(aa, 200)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 0, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 24, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 20, 64, 64)
db_d = core.neo_f3kdb.Deband(aa, 18, 64, 64, 64, 24, 0, sample_mode=4)
db_e = core.neo_f3kdb.Deband(aa, 16, 80, 80, 80, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db_f = core.std.MaskedMerge(db_c, db_e, adapt_m_2)
db = fvf.rfs(db, db_c, mappings="[1055 1084] [4933 5138] [13668 13727] [14775 14909] [18468 18566]")
db = fvf.rfs(db, db_d, mappings="[13866 14243] [14649 14774] [15840 16051] [18223 18406] [16611 16982] [17554 17781]")
db = fvf.rfs(db, db_f, mappings="[18091 18150]")
db = fvf.rfs(db, db_e, mappings="[30035 30108]")

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, var=1, constant=False)
grain = fvf.rfs(grain_a, grain_b, mappings="[5149 5598] [8691 10137]")

crop = core.std.Crop(grain, 240, 240, 0, 0)
borders = core.std.AddBorders(crop, 240, 240, 0, 0)
borders = fvf.rfs(grain, borders, mappings="[5149 5598]")

dering = hvf.HQDeringmod(db, mrad=1, msmooth=2, mthr=220, drrep=13, thr=42, planes=[0], show=False)
dering = fvf.rfs(borders, dering, mappings="[33927 34045]")

final = fvf.Depth(dering, 10)

final.set_output()