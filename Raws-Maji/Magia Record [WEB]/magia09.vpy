import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrainSF
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src_ = [
	lvf.src(r'WKN/Magia_Record_09_RU_HD_V2.mp4'),
	lvf.src(r'WKN/Magia Record - 09 (Wakanim SC 1080p).mkv'),
	lvf.src(r'WKN/Magia Record E09 [1080p][AAC][JapDub][GerSub][Web-DL].mkv'),
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 09 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 09 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 09 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_09_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33927)], asrc, 	"Magia09a_cut.mka")
ac.eztrim(src_amz, [(0,119)], 	ablank, "Magia09b_cut.mka")
O = 33940
ac.eztrim(src_[0],[(O,O+1)], 					asrc_wkn, "Magia09a_cut.mka")
ac.eztrim(src_[0],[(0,3140)], 					asrc_wkn, "Magia09b_cut.mka")
ac.eztrim(src_[0],[(O,O+1)], 					asrc_wkn, "Magia09c_cut.mka")
ac.eztrim(src_[0],[(3140,20978)],				asrc_wkn, "Magia09d_cut.mka")
ac.eztrim(src_[0],[(O,O+12)], 					asrc_wkn, "Magia09e_cut.mka")
ac.eztrim(src_[0],[(20978,33913),(33985,-7)], 	asrc_wkn, "Magia09f_cut.mka")

"""
src_ = [core.resize.Bicubic(src, kgf.get_w(486), 486) for src in src_]
src_ = [core.text.FrameNum(src, 4) for src in src_]
src_fu = core.resize.Bicubic(src_fu, kgf.get_w(486), 486)
src_fu = core.text.FrameNum(src_fu, 4)
src_amz = core.resize.Bicubic(src_amz, kgf.get_w(486), 486)
src_amz = core.text.FrameNum(src_amz, 4)

Infos = ["WKN RU", "WKN SC", "WKN DE", "WKN FR"]
for n, src in enumerate(src_):
	src_[n] = core.text.Text(src_[n], f"{Infos[n]}")

src_fu = core.text.Text(src_fu, "Funimation")
src_amz = core.text.Text(src_amz, "Amazon")
"""

# Slicing
def slicing(clip: vs.VideoNode, clip_op_ed: vs.VideoNode = None, clip_fu: vs.VideoNode = None) -> vs.VideoNode:
	return (
		clip_fu[289] + clip[:983] + 
		clip_op_ed[983:3140] + clip[3140] + clip[3140:20978] + 
		clip_fu[21269:21269+12] + clip[20978:31204] +
		clip_op_ed[31204:33361] + clip[33361:33913] + clip[33985:-7]
	)

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34216] + src_fu[34456:34576]
# src_amz = src_amz[:33927] + src_amz[34167:]

# Slicing comparison
# scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz), 
								# lvf.stack_compare(src_[1], src_[2], src_[3])])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
# src_amz = fvf.Depth(src_amz, 16)
src_ru = src_[0]
src_sc = src_[1]
src_de = src_[2]

hardsubmask = kgf.hardsubmask(src_de, src_ru)
src = core.std.MaskedMerge(src_de, src_ru, hardsubmask)

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)

src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=200)), 
	mappings="[17093 17134]")

src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=0)), 
	mappings="[19306 19485] [19756 19764] [22735 22773] [23455 23490]" +
	"[23776 23805]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), 
	mappings="[3177 3272] [3537 3554] [4545 4655] [18109 18259]" +
	"[33813 34046]")

# scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
								# lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])

edstart, edend = 31218, 33374
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

den_a = fvf.Depth(CoolDegrainSF(src, tr=2, thsad=48, blksize=8, overlap=4), 16)
den_b = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

den = core.knlm.KNLMeansCL(den, a=2, h=0.65, d=3, device_type='gpu', channels='UV')


aa = TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)

adapt_m = core.adg.Mask(aa.std.PlaneStats(), 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 17, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=24, preset='depth')
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[3612 3635] [4899 5042]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

endcard = lvf.upscaled_sraa(den, 2, sharp_downscale=False)
endcard = fvf.rfs(grain, endcard, mappings="[33927 34046]")

final = fvf.Depth(endcard, 10)

final.set_output()