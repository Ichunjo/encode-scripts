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

src_a = lvf.src(r'WKN/Magia Record - 01 (WEB-DL 1920x1080 x264 AAC Rus + Jap).mkv')
src_b = lvf.src(r'Funi/[HorribleSubs] Magia Record - 01 [1080p].mkv')[289:]
src_c = lvf.src(r'WKN/Magia_Record_01_FR_HD.mp4')

asrc = r"WKN/Magia_Record_01_FR_HD.mka"
ac.eztrim(src_a, [(0,1)], 								asrc, "Magia01a_cut.mka")
ac.eztrim(src_a, [(0,18150),(18154,33930),(33948,0)], 	asrc, "Magia01b_cut.mka")
ac.eztrim(src_a, [(0,12)], 								asrc, "Magia01c_cut.mka")

src_a = src_a[0]+src_a[:18150]+src_a[18154:33930]+src_a[33948:]+(src_a[-1]*12)
src_b = src_b[:-1]
src_c = src_c[0]+src_c[:18150]+src_c[18154:33930]+src_c[33948:]+(src_c[-1]*12)

vrdf.generate_keyframes(src_a, 'magia01_keyframes.log')

# scomp = lvf.stack_compare(src_a, src_b, src_c, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)
src_c = fvf.Depth(src_c, 16)

hardsubmask_ab = kgf.hardsubmask(src_a, src_b)
hardsubmask_ac = kgf.hardsubmask(src_a, src_c)
src_ab = core.std.MaskedMerge(src_a, src_b, hardsubmask_ab)
src_ac = core.std.MaskedMerge(src_a, src_c, hardsubmask_ac)

src = fvf.rfs(src_a, src_ab, mappings="[7127 7177]")
src = fvf.rfs(src, src_ac, mappings="[32341 33097]")

src_fade_ab = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade_ab_2 = kgf.hardsubmask_fades(src_a, src_b, highpass=600)
src_fade_ab_3 = kgf.hardsubmask_fades(src_a, src_b, highpass=1300)
src_fade_ac = kgf.hardsubmask_fades(src_a, src_c, highpass=2000)

src_fade_ab = core.std.MaskedMerge(src_a, src_b, src_fade_ab)
src_fade_ab_2 = core.std.MaskedMerge(src_a, src_b, src_fade_ab_2)
src_fade_ab_3 = core.std.MaskedMerge(src_a, src_b, src_fade_ab_3)

src_fade_ac_1 = core.std.MaskedMerge(src_a, src_c, src_fade_ac)
src_fade_ac_2 = core.std.MaskedMerge(src_a, src_c, vrdf.RegionMask(src_fade_ac, left=211, top=407, right=155, bottom=620))
src_fade_ac_3 = core.std.MaskedMerge(src_a, src_c, vrdf.RegionMask(src_fade_ac, left=547, top=253, right=541, bottom=683))

src = fvf.rfs(src, src_fade_ab, mappings="[13398 13451] [15150 15491] [25512 25652] [30361 30844] [31649 31768] [32320 32340] [33927 34045]")
src = fvf.rfs(src, src_fade_ab_2, mappings="[10525 10726]")
src = fvf.rfs(src, src_fade_ab_3, mappings="[32083 32284]")
src = fvf.rfs(src, src_fade_ac_1, mappings="[33098 33112]")
src = fvf.rfs(src, src_fade_ac_2, mappings="[9252 9749]")
src = fvf.rfs(src, src_fade_ac_3, mappings="[13518 13583]")

#scomp2 = lvf.stack_compare(src, src_b, src_c, height=540)

den_a = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(src, tr=1, thsad=168, thsadc=120, bits=16, blksize=8, overlap=4)
den = fvf.rfs(den_a, den_b, mappings="[30035 30108]")

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, down8=False, opencl=True)
aa_b = lvf.upscaled_sraa(den, rep=6, sharp_downscale=False)
aa = fvf.rfs(aa_a, aa_b, mappings="[29453 29476] [29510 29532] [29640 29663] [29775 29798] [29866 29889] [30011 30034] [30211 30282]")

adapt_m = vrdf.adaptive_mask(aa, 28)
adapt_m_2 = vrdf.adaptive_mask(aa, 200)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 20, 64, 64)
db_d = core.neo_f3kdb.Deband(aa, 18, 64, 64, 64, 20, 0, sample_mode=4)
db_e = core.neo_f3kdb.Deband(aa, 16, 80, 80, 80, 20, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db_f = core.std.MaskedMerge(db_c, db_e, adapt_m_2)
db_g = core.neo_f3kdb.Deband(db_e, 20, 36, 0, 0, 0, 0, sample_mode=4)
db = fvf.rfs(db, db_c, mappings="[1055 1084] [4933 5138] [13668 13727] [14775 14909] [18468 18566]")
db = fvf.rfs(db, db_d, mappings="[13866 14243] [14649 14774] [15840 16051] [18223 18406] [16611 16982] [17554 17781]")
db = fvf.rfs(db, db_f, mappings="[18091 18150]")
db = fvf.rfs(db, db_e, mappings="[30035 30108]")
db = fvf.rfs(db, db_g, mappings="[30211 30282]")

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, var=1, constant=False)
grain_c = kgf.adaptive_grain(db, 0.6, False, 20)
grain = fvf.rfs(grain_a, grain_b, mappings="[5149 5598] [8691 10137]")
grain = fvf.rfs(grain, grain_c, mappings="[30035 30108]")

crop = core.std.Crop(grain, 240, 240, 0, 0)
borders = core.std.AddBorders(crop, 240, 240, 0, 0)
borders = fvf.rfs(grain, borders, mappings="[5149 5598]")

dering = hvf.HQDeringmod(db, mrad=1, msmooth=2, mthr=220, drrep=13, thr=42, planes=[0], show=False)
dering = fvf.rfs(borders, dering, mappings="[33927 34045]")

final = fvf.Depth(dering, 10)

final.set_output()