import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrain
from adjust import Tweak

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src_ = [
    lvf.src(r'WKN/Magia_Record_13_RU_HD_V2.mp4'),
    lvf.src(r'WKN/Magia Record - 13v2 (Wakanim SC 1080p).mkv'),
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 13 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 13 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 13 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_13_FR_HD_V2.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33927)], asrc, "Magia13a_cut.mka")
ac.eztrim(src_amz, [(0,118)], 	ablank, "Magia13b_cut.mka")
O = 33940
ac.eztrim(src_[0],[(O,O+1)], 				asrc_wkn, "Magia13a_cut.mka")
ac.eztrim(src_[0],[(0,33926),(33944,-15)], 	asrc_wkn, "Magia13b_cut.mka")

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
        clip_fu[289] + clip[:3260] + 
        clip_op_ed[3260:5416] + clip_op_ed[5415] * 2 + clip[5418:32128] +
        clip_op_ed[32128:33098] + clip[33098:33926] +
        clip[33944:-15]
    )

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34216] + src_fu[34456:34576]
#src_amz = src_amz[:33927] + src_amz[34167:]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.scomp(src_[0], src_fu, src_amz, make_diff=False, warn=False), 
#								lvf.scomp(src_[1], src_[2], src_[3], make_diff=False, warn=False)])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
src = src_[0]
src_sc = src_[1]

src_fade = kgf.hardsubmask_fades(src, src_sc, expand_n=28, highpass=600)
    
    
src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
    vrdf.region_mask(src_fade, left=0, right=0, top=0, bottom=200)),
    mappings="[33275 33276] [33377 33398] [33417 33444] [33463 33503]" +
    "[33532 33533] [33583 33584] [33617 33655]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade),
    mappings="[9787 9834] [9859 9930] [13514 13561] [33099 33206]" +
    "[33300 33317] [33348 33365] [33399 33416] [33445 33462] [33504 33531]" +
    "[33683 33691] [33713 33714] [33927 34046]")

#scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486, make_diff=False, warn=False),
#								lvf.stack_compare(src_sc, src_[2], src_[3], height=486, make_diff=False, warn=False)])


dimming = Tweak(fvf.Depth(src, 32), sat=1.85, cont=1.85)
dimming = fvf.rfs(src, fvf.Depth(dimming, 16), mappings="[33318 33347] [33668 33712]")

den = CoolDegrain(dimming, tr=2, thsad=48, blksize=8, overlap=4)

aa_a = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)
aa_b = taa.TAAmbk(aa_a, aatype='Eedi3', cycle=2)
aa = fvf.rfs(aa_a, aa_b, mappings="[16914 17000]")

adapt_m = core.adg.Mask(aa.std.PlaneStats(), 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 17, 64, 64)
db_c = core.neo_f3kdb.Deband(db_c, grainy=24, preset='depth')
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[5614 5760] [6017 6094] [6344 6454] [6632 6787] [7199 7246] [16601 16766] [33720 33732]")

grain = kgf.adaptive_grain(db, 0.25)

endcard = fvf.rfs(grain, den, mappings="[33927 34046]")
ending = fvf.rfs(endcard, src, mappings="[32079 33098]")

final = fvf.Depth(ending, 10)
final.set_output()