import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrainSF

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src_ = [
    lvf.src(r'WKN/Magia_Record_10_RU_HD.mp4'),
    lvf.src(r'WKN/Magia Record - 10 (Wakanim SC 1080p).mkv'),
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 10 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 10 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 10 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_10_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33926)], asrc, 	"Magia10a_cut.mka")
ac.eztrim(src_amz, [(0,118)], 	ablank, "Magia10b_cut.mka")
O = 33940
ac.eztrim(src_[0],[(O,O+1)], 						asrc_wkn, "Magia10a_cut.mka")
ac.eztrim(src_[0],[(0,4769)], 						asrc_wkn, "Magia10b_cut.mka")
ac.eztrim(src_[0],[(O,O+1)], 						asrc_wkn, "Magia10c_cut.mka")
ac.eztrim(src_[0],[(4769,16638),(16651+2,33937)], 	asrc_wkn, "Magia10d_cut.mka")
ac.eztrim(src_[0],[(O,O+2)], 						asrc_wkn, "Magia10e_cut.mka")
ac.eztrim(src_[0],[(33937,-8)], 					asrc_wkn, "Magia10f_cut.mka")

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
        clip_fu[289] + clip[:2613] +
        clip_op_ed[2613:4769] + clip_op_ed[4768] + 
        clip[4769:16638] + clip[16651+2:28042] + 
        clip_op_ed[28042:30318] + clip[30318:33937] + clip[33936]*2 +  clip[33937:-8]
    )

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34215] + src_fu[34455:34575]
#src_amz = src_amz[:33926] + src_amz[34166:]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz), 
#								lvf.stack_compare(src_[1], src_[2], src_[3])])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
src = src_[0]
src_sc = src_[1]

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)
    
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, src_fade), 
    mappings="[9481 9518]")	

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), 
    mappings="[5278 5301] [5836 5910] [6006 6053] [6576 6638]" +
    "[28029 28148] [33926 34045]")

#scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
#								lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])

edstart, edend = 28149, 30304
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

den_a = fvf.Depth(CoolDegrainSF(src, tr=2, thsad=48, blksize=8, overlap=4), 16)
den_b = vrdf.knlmcl(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)

adapt_m = core.adg.Mask(aa.std.PlaneStats(), 28)
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

endcard = fvf.rfs(grain, den, mappings="[33926 34045]")

final = fvf.Depth(endcard, 10)

final.set_output()