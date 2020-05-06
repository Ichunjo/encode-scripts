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
    lvf.src(r'WKN/Magia_Record_11_RU_HD.mp4'),
    lvf.src(r'WKN/Magia Record - 11 (Wakanim SC 1080p).mkv'),
    lvf.src(r'WKN/Magia Record E11 [1080p][AAC][JapDub][GerSub][Web-DL].mkv'),
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 11 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 11 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 11 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_11_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,33927)], asrc, 	"Magia11a_cut.mka")
ac.eztrim(src_amz, [(0,118)], 	ablank, "Magia11b_cut.mka")
O = 33940
ac.eztrim(src_[0],[(O,O+1)], 					asrc_wkn, "Magia11a_cut.mka")
ac.eztrim(src_[0],[(0,3522)], 					asrc_wkn, "Magia11b_cut.mka")
ac.eztrim(src_[0],[(O,O+9)], 					asrc_wkn, "Magia11c_cut.mka")
ac.eztrim(src_[0],[(3522,14376)], 				asrc_wkn, "Magia11d_cut.mka")
ac.eztrim(src_[0],[(O,O+8)], 					asrc_wkn, "Magia11e_cut.mka")
ac.eztrim(src_[0],[(14393,33926),(33929,-8)], 	asrc_wkn, "Magia11f_cut.mka")

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
        clip[0] + clip[:1367] + 
        clip_op_ed[1367:3522] + clip_op_ed[3521]*2 + clip[3522]*7 +
        clip[3522:14376] + clip[14393]*8 + clip[14393:31769] + 
        clip_op_ed[31769:33926] + clip[33929:-8]
    )

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34216] + src_fu[34456:34576]
#src_amz = src_amz[:33927] + src_amz[34167:]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.stack_compare(src_[0], src_fu, src_amz), 
#								lvf.stack_compare(src_[1], src_[2], src_[3])])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
src = src_[0]
src_sc = src_[1]
src_de = src_[2]

src_fade = kgf.hardsubmask_fades(src, src_sc, highpass=2000)

src = fvf.rfs(src, core.std.MaskedMerge(src, src_sc, 
    vrdf.region_mask(src_fade, left=0, right=0, top=0, bottom=200)), 
    mappings="[16625 16671] [25756 25827]")

src_fade = kgf.hardsubmask_fades(src, src_de, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_de, 
    vrdf.region_mask(src_fade, left=0, right=0, top=0, bottom=200)), 
    mappings="[24369 24395]")

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), 
    mappings="[96 191] [6449 6517] [7790 7930] [8303 8374] [10475 10570]" +
    "[13858 14124] [18713 18751] [22369 22434] [25708 25755] [31649 31768] [33927 34046]")

#scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486),
#								lvf.stack_compare(src_sc, src_[2], src_[3], height=486)])

edstart, edend = 31769, 33926
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

mask_LightArrows = core.std.Expr([
    vrdf.get_y(lvf.src('mask for light.png', aa)),
    lvf.denoise.detail_mask(den_a, brz_a=2000, brz_b=1500)],
    'x y max')
db_LightArrows = core.placebo.Deband(aa, radius=12, threshold=6, iterations=1, grain=0, planes=1)
db_LightArrows = core.grain.Add(db_LightArrows, 0.25, constant=True)
db_LightArrows = core.std.MaskedMerge(db_LightArrows, db, mask_LightArrows)

db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)
db = fvf.rfs(db, db_LightArrows, mappings='[19018 19089]')

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

endcard = fvf.rfs(grain, den, mappings="[33927 34046]")

final = fvf.Depth(endcard, 10)

final.set_output()
