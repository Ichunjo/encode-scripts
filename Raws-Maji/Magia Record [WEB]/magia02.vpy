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

src_a = lvf.src(r'WKN/Magia Record E02 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'Funi/[HorribleSubs] Magia Record - 02 [1080p].mkv')[289:]

asrc = r"WKN/Magia_Record_02_FR_HD.mka"
ac.eztrim(src_a, [(0,1)], 				asrc, "Magia02a_cut.mka")
ac.eztrim(src_a, [(0,4456)], 			asrc, "Magia02b_cut.mka")
ac.eztrim(src_a, [(0,4),(4456,16875)], 	asrc, "Magia02c_cut.mka")
ac.eztrim(src_a, [(0,10),(16878,0)], 	asrc, "Magia02d_cut.mka")
ac.eztrim(src_a, [(0,21)], 				asrc, "Magia02e_cut.mka")

src_a = src_b[0]+src_a[:4456]+(src_a[4455]*2)+src_b[4459:4461]+src_a[4456:16875]+(src_a[16878]*10)+src_a[16878:]+(src_a[-1]*21)

vrdf.generate_keyframes(src_a, 'magia02_keyframes.log')

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade_a = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade_b = kgf.hardsubmask_fades(src_a, src_b, highpass=1300)
src_fade_a = core.std.MaskedMerge(src_a, src_b, src_fade_a)
src_fade_b = core.std.MaskedMerge(src_a, src_b, src_fade_b)

square_m = kgf.squaremask(src, 792, 117, 552, 43).std.Inflate()
src_fade_c = core.std.MaskedMerge(src_a, src_b, square_m)

src = fvf.rfs(src, src_fade_a, mappings="[5288 5479] [5822 5926] [6035 6106] [6326 6505] [16976 17124] [17362 17415] [22736 22839] [24580 24795] [31481 31767] [33926 34045] [17163 17199]")
src = fvf.rfs(src, src_fade_b, mappings="[2618 2815] [4390 4458]")
src = fvf.rfs(src, src_fade_c, mappings="[24220 24267]")

#scomp2 = lvf.stack_compare(src, src_b, fvf.Depth(src_c, 16), height=540)

edstart = 31768
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

adapt_m = vrdf.adaptive_mask(src, 10).std.Invert()
den_a = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(src, tr=1, thsad=168, thsadc=192, bits=16, blksize=8, overlap=4)
den_b = core.std.MaskedMerge(den_a, den_b, adapt_m)
den_c = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings="[7986 10966]")
den = fvf.rfs(den, den_c, mappings=ending_a + ending_b)

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False, opencl=False)

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
db = fvf.rfs(db, db_c, mappings="[22494 22616]")
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