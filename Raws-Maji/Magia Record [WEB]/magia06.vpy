import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrainSF

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src_ = [
	lvf.src(r'WKN/Magia_Record_06_RU_HD.mp4'),
	lvf.src(r'WKN/Magia Record - 06 (Wakanim SC 1080p).mkv'),
	lvf.src(r'WKN/Magia Record - 06 (Wakanim DE 1080p).mkv'),
	lvf.src(r'WKN/Magia_Record_06_FR_HD.mp4')
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 06 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 06 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 06 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_06_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33927)], asrc, 	"Magia06a_cut.mka")
ac.eztrim(src_amz, [(0,118)], 	ablank, "Magia06b_cut.mka")
O = 33928
ac.eztrim(src_[3],[(0,1)], 			asrc_wkn, "Magia06a_cut.mka")
ac.eztrim(src_[3],[(0,4003)], 		asrc_wkn, "Magia06b_cut.mka")
ac.eztrim(src_[3],[(O,O+13)], 		asrc_wkn, "Magia06c_cut.mka")
ac.eztrim(src_[3],[(4003,15258)], 	asrc_wkn, "Magia06d_cut.mka")
ac.eztrim(src_[3],[(O,O+2)], 		asrc_wkn, "Magia06e_cut.mka")
ac.eztrim(src_[3],[(15258,-7)], 	asrc_wkn, "Magia06f_cut.mka")

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
		clip[0] + clip[:1845] +
		clip_op_ed[1845:4003] + 
		clip_op_ed[4002] + clip[4003] * 12 + clip[4003:15258] + clip_fu[15561:15563] + clip[15258:31753] +
		clip_op_ed[31753:33911] + clip[33911:-9]
	)

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34216] + src_fu[34456:34576]
# src_amz = src_amz[:33927] + src_amz[34167:-2]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz), 
#								lvf.stack_compare(src_[1], src_[2], src_[3])])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
# src_amz = fvf.Depth(src_amz, 16)
src = src_[0]
src_sc = src_[1]
src_de = src_[2]

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)
	
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=0)), 
	mappings="[7147 7176] [13888 13998]")
	
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=200)), 
	mappings="[7351 7389] [16254 16409] [22372 22550] [24884 24929]")

src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=200)), 
	mappings="[7910 8004] [8389 8505] [9301 9360]")

src = fvf.rfs(src, src_sc, mappings="[7630 7698] [15274 15321] [15037 15117]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), mappings=
	"[7210 7242] [7390 7474] [12775 12822] [13591 13617] [17601 17651]" +
	"[17694 17855] [18189 18212] [20022 20057] [31661 31768] [33927 34045]")

scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
								lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])


edstart, edend = 31769, 33926
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

deint = core.eedi3m.EEDI3CL(src, 1)
src = fvf.rfs(src, deint, mappings='[8185 8280]')

den_a = fvf.Depth(CoolDegrainSF(src, tr=2, thsad=48, blksize=8, overlap=4), 16)
den_b = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)
aa_b = taa.TAAmbk(den, aatype='Eedi3')
aa = fvf.rfs(aa_a, aa_b, mappings="[8185 8280]")

adapt_m = vrdf.adaptive_mask(aa, 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 17, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=24, preset='depth')
db_d = core.neo_f3kdb.Deband(aa, 17, 80, 60, 64, 20, 0, sample_mode=4)
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[10218 10262] [17415 17519] [20022 20057]")
db = fvf.rfs(db, db_d, mappings="[9852 9920]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

endcard = lvf.upscaled_sraa(den, 2, sharp_downscale=False)
endcard = fvf.rfs(grain, endcard, mappings="[33927 34045]")

final = fvf.Depth(endcard, 10)

final[15037:15196].set_output()
#src.set_output()