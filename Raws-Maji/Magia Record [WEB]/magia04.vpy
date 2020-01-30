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

# Wakanim
src_ = [
	lvf.src(r'WKN/Magia_Record_04_RU_HD.mp4'),
	lvf.src(r'WKN/Magia Record - 04 (Wakanim SC 1080p).mkv'),
	lvf.src(r'WKN/Magia Record - 04 (Wakanim DE 1080p).mkv'),
	# lvf.src(r'WKN/Magia_Record_04_FR_HD.mp4'),
]

# Other raws
src_fu = lvf.src(r'Funi/[HorribleSubs] Magia Record - 04 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 04 (Amazon dAnime VBR 720p).mkv')
src_bs11 = lvf.src(r'TV/Magia Record - 04 (BS11).d2v')
src_bs11 = core.vivtc.VDecimate(src_bs11)

# Audio cut
asrc = 		r"AMZ/Magia Record - 04 (Amazon dAnime VBR 720p).mka"
ablank = 	r"blank.eac3"
asrc_bs11 = r"TV/Magia Record - 04 (BS11) T141 stereo 248 kbps DELAY -366 ms.mka"
ac.eztrim(src_amz, 	[(0,33926)], 								asrc, 		"Magia04a_cut.mka")
ac.eztrim(src_amz, 	[(0,119)], 									ablank, 	"Magia04b_cut.mka")
ac.eztrim(src_bs11, [(685,3753),(5432,14160),(15597,37728)], 	asrc_bs11, 	"TV/Magia04_bs11_cut.mka")

"""
src_fu = core.text.Text(src_fu, 'Funimation')
src_[0] = core.text.Text(src_[0], 'WKN RU')
src_[1] = core.text.Text(src_[1], 'WKN SC')
src_[2] = core.text.Text(src_[2], 'WKN DE')
src_[3] = core.text.Text(src_[3], 'WKN FR')
"""

"""
src_ = [core.text.FrameNum(src) for src in src_]
src_fu = core.text.FrameNum(src_fu)
"""

# Slicing
def slicing(clip: vs.VideoNode, clip_op_ed: vs.VideoNode = None, clip_fu: vs.VideoNode = None) -> vs.VideoNode:
	return (
		clip[0] + clip[:912] +
		clip_op_ed[912:3067] +
		clip_op_ed[3066] + clip[3067:11781] + clip[11780] * 13 + clip_fu[11796:11796+7] + clip[11781:31746] + 
		clip_op_ed[31746:33904] + 
		clip[33904:34024]
	)

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu[289:34335]
src_fu = src_fu + src_fu[-1]

# Slicing comparison
# scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz, height=720),
								# lvf.stack_compare(src_[1], src_[2], src_[3], height=720)])

# Variables & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
src = src_[0]
src_sc = src_[1]
src_de = src_[2]

src_fade = kgf.hardsubmask_fades(src, src_sc, expand_n=10, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=200)), mappings=
	"[6793 6828] [7385 7407] [7546 7594] [9420 9587] [11796 11879] [15632 15684]" +
	"[15977 16015] [21987 22058] [23149 23241] [24285 24331] [24484 24591] [27370 27450]" +
	"[678 720]")
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=0)), 
	mappings="[3780 3827] [15302 15412] [16016 16189]")
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=200, bottom=200)), 
	mappings="[4764 4984] [12199 12328] [22059 22110]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), mappings=
	"[0 47] [68 131] [216 346] [751 816] [852 911] [3297 3368] [3453 3497] [3558 3605]" + 
	"[3648 3695] [6775 6792] [7498 7545] [8027 8067] [9588 9647] [14162 14764] [17111 17158]" +
	"[17709 17750] [18306 18356] [22301 22342] [23455 23480] [23884 23906] [24332 24363]" + 
	"[27451 27507] [28500 28535] [28536 28658] [29668 29760] [33926 34045] [16358 16408]" +
	"[23149 23241] [31648 31767]")

src = fvf.rfs(src, src_de, mappings="[21833 21862] [21955 21986]")

src_fade = kgf.hardsubmask_fades(src, src_de, expand_n=10, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_de, 
	vrdf.RegionMask(src_fade, left=0, right=0, top=0, bottom=200)), 
	mappings="[21923 21954]")

#scomp = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
								#lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])

edstart, edend = 31768, 33925
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

den_a = CoolDegrain(src, tr=1, thsad=84, thsadc=60, bits=16, blksize=8, overlap=4)
den_b = vrdf.KNLMCL(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)
aa_b = taa.TAAmbk(den, aatype='Eedi3')
aa = fvf.rfs(aa_a, aa_b, mappings="[19889 20008]")

adapt_m = vrdf.adaptive_mask(aa, 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

crop = core.std.CropRel(grain, left=240, top=0, right=240, bottom=0)
bd = core.std.AddBorders(crop, left=240, top=0, right=240, bottom=0)
bd = fvf.rfs(grain, bd, mappings="[18524 18547] [18673 18715] [18875 18967] [19817 19888]")

endcard = lvf.upscaled_sraa(den, 2, sharp_downscale=False)
endcard = fvf.rfs(bd, endcard, mappings="[33926 34045]")

final = fvf.Depth(endcard, 10)

final.set_output()