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
core.max_cache_size = 26000

ac = acsuite.AC()

src_ = [
	core.dgdecodenv.DGSource(r'WKN/Magia_Record_05_RU_HD.dgi'),
	core.dgdecodenv.DGSource(r'WKN/Magia Record - 05 (Wakanim SC 1080p).dgi'),
	core.dgdecodenv.DGSource(r'WKN/Magia Record E05 [1080p][AAC][JapDub][GerSub][Web-DL].dgi'),
	core.dgdecodenv.DGSource(r'WKN/Magia_Record_05_FR_HD.dgi')
]

src_fu = core.dgdecodenv.DGSource(r'FUNI/[HorribleSubs] Magia Record - 05 [1080p].dgi')[289:34338]
src_amz = core.dgdecodenv.DGSource(r'AMZ/Magia Record - 05 (Amazon dAnime VBR 720p).dgi')

asrc = 		r"AMZ/Magia Record - 05 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_05_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33929)], asrc, 	"Magia05a_cut.mka")
ac.eztrim(src_amz, [(0,119)], 	ablank, "Magia05b_cut.mka")
O = 33910
ac.eztrim(src_[3],[(0,1)], 			 asrc_wkn, "Magia05a_cut.mka")
ac.eztrim(src_[3],[(0,4242)], 		 asrc_wkn, "Magia05b_cut.mka")
ac.eztrim(src_[3],[(0,2)], 			 asrc_wkn, "Magia05c_cut.mka")
ac.eztrim(src_[3],[(4242,18366)], 	 asrc_wkn, "Magia05d_cut.mka")
ac.eztrim(src_[3],[(O,O+16)], 		 asrc_wkn, "Magia05e_cut.mka")
ac.eztrim(src_[3],[(18368,33910+2)], asrc_wkn, "Magia05f_cut.mka")
ac.eztrim(src_[3],[(33910,34028)], 	 asrc_wkn, "Magia05g_cut.mka")

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
		clip[0] + clip[:2085] +
		clip_op_ed[2085:4242] +
		clip_op_ed[4241] * 2 + clip[4242:18366] + clip[18368] * 16 + clip[18368:29357] +
		clip_op_ed[29357:31514] + 
		clip[31514:33910] + clip_fu[33927:33929] + clip[33910:34028]
	)

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu[:34047]
#src_amz = src_amz[:33929] + src_amz[34169:]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz), 
#								lvf.stack_compare(src_[1], src_[2], src_[3])])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
#src_amz = fvf.Depth(src_amz, 16)
src = src_[0]
src_sc = src_[1]
src_de = src_[2]

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, src_fade), 
	mappings="[795 854]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), mappings=
	"[24305 24328] [24368 24379] [24422 24433] [24470 24481] [29265 29373]" +
	"[33929 34046]")

#scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
#								lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])

edstart, edend = 29374, 31530
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

den_a = fvf.Depth(CoolDegrainSF(src, tr=2, thsad=48, blksize=8, overlap=4), 16)
den_b = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)
aa_b = taa.TAAmbk(den, aatype='Eedi3')
aa = fvf.rfs(aa_a, aa_b, mappings="[13611 13661]")

adapt_m = vrdf.adaptive_mask(aa, 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 17, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=24, preset='depth')
db_d = core.neo_f3kdb.Deband(aa, 17, 80, 60, 64, 20, 0, sample_mode=4)
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[5465 5524] [5792 5836] [5882 5955] [13611 13661] [13742 13837] [13955 13990] [25202 25279]")
db = fvf.rfs(db, db_d, mappings="[13542 13610]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

crop = core.std.CropRel(grain, left=240, top=0, right=240, bottom=0)
bd = core.std.AddBorders(crop, left=240, top=0, right=240, bottom=0)
bd = fvf.rfs(grain, bd, mappings="[0 707] [7317 7364] [7506 7559] [7695 7724]")

endcard = lvf.upscaled_sraa(den, 2, sharp_downscale=False)
endcard = fvf.rfs(bd, endcard, mappings="[33929 34046]")

final = fvf.Depth(endcard, 10)

final.set_output()