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

src_ru = lvf.src(r'WKN/Magia_Record_02_RU_HD.mp4')
src_sc = lvf.src(r'WKN/Magia Record - 02 (Wakanim SC 1080p).mkv')
src_fu = lvf.src(r'Funi/[HorribleSubs] Magia Record - 02 [1080p].mkv')[289:]

asrc = r"WKN/Magia_Record_02_RU_HD.mka"
ac.eztrim(src_ru, [(0,1)], 				asrc, "Magia02a_cut.mka")
ac.eztrim(src_ru, [(0,4456)], 			asrc, "Magia02b_cut.mka")
ac.eztrim(src_ru, [(0,4),(4456,16875)], asrc, "Magia02c_cut.mka")
ac.eztrim(src_ru, [(0,10),(16878,0)], 	asrc, "Magia02d_cut.mka")
ac.eztrim(src_ru, [(0,21)], 			asrc, "Magia02e_cut.mka")

src_ru = src_fu[0]+src_ru[:4456]+(src_ru[4455]*2)+src_fu[4459:4461]+src_ru[4456:16875]+(src_ru[16878]*10)+src_ru[16878:]+(src_ru[-1]*21)
src_sc = src_fu[0]+src_sc[:4456]+(src_sc[4455]*2)+src_fu[4459:4461]+src_sc[4456:16875]+(src_sc[16878]*10)+src_sc[16878:]+(src_sc[-1]*21)

vrdf.generate_keyframes(src_ru, 'magia02_keyframes.log')

opstart, opend = 2302, 4458
edstart, edend = 31768, 33925

src_ru = fvf.Depth(src_ru, 16)
src_sc = fvf.Depth(src_sc, 16)
src_fu = fvf.Depth(src_fu, 16)

src = src_ru[:opstart]+src_sc[opstart:opend+1]+src_ru[opend+1:edstart]+src_sc[edstart:edend+1]+src_ru[edend+1:]

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)
src_fade = vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=237)
src_fade = core.std.MaskedMerge(src, src_sc, src_fade)
src = fvf.rfs(src, src_fade, mappings="[5288 5479] [5822 5926] [6326 6367] [16976 17071] [22736 22839] [24580 24735] [31481 31665]")

src_fade = kgf.hardsubmask_fades(src, src_fu, highpass=2000)
src_fade = core.std.MaskedMerge(src, src_fu, src_fade)
src = fvf.rfs(src, src_fade, mappings="[6035 6106] [6368 6505] [17072 17124] [17163 17199] [17362 17415] [24736 24795] [26137 26172] [31666 31767] [33926 34045]")

src_fade = kgf.hardsubmask_fades(src_sc, src_fu, highpass=2000)
src_fade = core.std.MaskedMerge(src_sc, src_fu, src_fade)
src = fvf.rfs(src, src_fade, mappings="[24220 24267]")

#scomp1 = lvf.stack_compare(src, src_sc, src_de, src_fr, src_fu, height=486)

ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

adapt_m = vrdf.adaptive_mask(src, 10).std.Invert()
den_a = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(src, tr=1, thsad=168, thsadc=192, bits=16, blksize=8, overlap=4)
den_b = core.std.MaskedMerge(den_a, den_b, adapt_m)
den_c = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings="[7986 10966]")
den = fvf.rfs(den, den_c, mappings=ending_a + ending_b)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)
aa_b = taa.TAAmbk(den, aatype='Eedi3', down8=False)
aa = fvf.rfs(aa_a, aa_b, mappings="[27742 27770]")

adapt_m = vrdf.adaptive_mask(aa, 28)
adapt_m_2 = vrdf.adaptive_mask(aa, 200)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 20, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=24, preset='depth')
db_d = core.std.MaskedMerge(db_b, db_c, adapt_m_2)
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[21294 21323] [22494 22616] [27679 27741] [27793 27810] [28364 28480]")
db = fvf.rfs(db, db_d, mappings="[7137 7270] [7355 7414] [25582 25695]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

endcard = fvf.rfs(grain, den, mappings="[33926 34045]")

final = fvf.Depth(endcard, 10)

final.set_output()