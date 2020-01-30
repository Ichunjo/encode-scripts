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

src_ru = lvf.src(r'WKN/Magia_Record_03_RU_HD.mp4')
src_sc = lvf.src(r'WKN/Magia Record - 03 (Wakanim SC 1080p).mkv')
src_de = lvf.src(r'WKN/Magia Record E03 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_fu = lvf.src(r'Funi/[HorribleSubs] Magia Record - 03 [1080p].mkv')[289:34336]

asrc = r"WKN/Magia Record E03 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
O = 33891
ac.eztrim(src_de, [(O,O+1)], 					asrc, "Magia03a_cut.mka")
ac.eztrim(src_de, [(0,3643)], 					asrc, "Magia03b_cut.mka")
ac.eztrim(src_de, [(O,O+13)], 					asrc, "Magia03c_cut.mka")
ac.eztrim(src_de, [(3643,18009),(O,O+8+13)], 	asrc, "Magia03d_cut.mka")
ac.eztrim(src_de, [(18009,33890),(O,O+2)], 		asrc, "Magia03e_cut.mka")
ac.eztrim(src_de, [(33890,-10)], 				asrc, "Magia03f_cut.mka")

src_ru = src_fu[0]+src_ru[:3643]+src_ru[3642]+src_ru[3643]*12+src_ru[3643:18009]+src_fu[18023:18031]+src_ru[18009]*13+src_ru[18009:33890]+src_ru[33889]*2+src_ru[33890:-10]
src_sc = src_fu[0]+src_sc[:3643]+src_sc[3642]+src_sc[3643]*12+src_sc[3643:18009]+src_fu[18023:18031]+src_sc[18009]*13+src_sc[18009:33890]+src_sc[33889]*2+src_sc[33890:-10]
src_de = src_fu[0]+src_de[:3643]+src_de[3642]+src_de[3643]*12+src_de[3643:18009]+src_fu[18023:18031]+src_de[18009]*13+src_de[18009:33890]+src_de[33889]*2+src_de[33890:-10]

vrdf.generate_keyframes(src_ru, 'magia03_keyframes.log')

#scomp = lvf.stack_compare(src, src_sc, src_de, src_fu, height=486)

edstart, edend = 29852, 32008

src_ru = fvf.Depth(src_ru, 16)
src_sc = fvf.Depth(src_sc, 16)
src_de = fvf.Depth(src_de, 16)
src_fu = fvf.Depth(src_fu, 16)

src = fvf.rfs(src_ru, src_sc, mappings=f'[{edstart} {edend}]')

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=236)), mappings="[3645 3698] [10191 10220] [32684 32764]")
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=0)), mappings="[20929 20952]")

src_fade = kgf.hardsubmask_fades(src, src_fu, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), mappings="[3534 3644] [4261 4341] [6031 6117] [8353 8400] [9652 9687] [9748 9789] [10221 10274] [17225 17242] [20080 20127] [23539 23610] [29673 29851] [33927 34046]")

src = fvf.rfs(src, src_de, mappings="[21100 21123] [21310 21321]")

#scomp1 = lvf.stack_compare(src, src_sc, src_de, src_fu, height=486)

ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

adapt_m = vrdf.adaptive_mask(src, 10).std.Invert()
den_a = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)
den_b = vrdf.KNLMCL(src, 1, 1, 0, 16)
den_c = CoolDegrain(src, tr=1, thsad=24, thsadc=60, bits=16, blksize=8, overlap=4)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)
den = fvf.rfs(den_a, den_c, mappings='[4726 5001] [16079 16294] [18478 19581] [19756 21699]')

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
db_g = core.neo_f3kdb.Deband(aa, 12, 260, 260, 260, 20, 0, sample_mode=4)
db_h = core.neo_f3kdb.Deband(aa, 14, 24, 24, 24, 0, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[11772 11797] [11961 12143] [12483 12506] [12537 12627] [12651 12680] [12813 12855] [12987 13016] [13059 13100]")
db = fvf.rfs(db, db_d, mappings="[8968 9015] [10881 11024]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)
db = fvf.rfs(db, db_g, mappings='[12507 12520]')
db = fvf.rfs(db, db_h, mappings='[18031 18315]')

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

crop = core.std.Crop(grain, 0, 0, 132, 132)
fe = core.edgefixer.ContinuityFixer(crop, 0, [1,0,0], 0, 0)
bd = core.std.AddBorders(fe, 0, 0, 132, 132)
bd = fvf.rfs(grain, bd, mappings="[0 2905]")

endcard = lvf.upscaled_sraa(den, 2, sharp_downscale=False)
endcard = fvf.rfs(bd, endcard, mappings="[33927 34046]")

final = fvf.Depth(endcard, 10)

final.set_output()