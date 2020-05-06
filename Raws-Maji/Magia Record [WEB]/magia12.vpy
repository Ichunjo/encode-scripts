import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import debandshit as dbs
import acsuite
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src_ = [
    lvf.src(r'WKN/Magia_Record_12_RU_HD.mp4'),
    lvf.src(r'WKN/Magia Record - 12 (Wakanim SC 1080p).mkv'),
]

src_fu = lvf.src(r'FUNI/[HorribleSubs] Magia Record - 12 [1080p].mkv')
src_amz = lvf.src(r'AMZ/Magia Record - 12 (Amazon dAnime VBR 720p).mkv')

asrc = 		r"AMZ/Magia Record - 12 (Amazon dAnime VBR 720p).mka"
asrc_wkn = 	r"WKN/Magia_Record_12_FR_HD.mka"
ablank = 	r"blank.eac3"
ac.eztrim(src_amz, [(0,7146),(7147,12279),(12280,33930)], asrc, "Magia12a_cut.mka")
ac.eztrim(src_amz, [(0,118)], 	ablank, "Magia12b_cut.mka")
O = 33940
ac.eztrim(src_[0],[(O,O+1)], 		asrc_wkn, "Magia12a_cut.mka")
ac.eztrim(src_[0],[(0,7144)], 		asrc_wkn, "Magia12b_cut.mka")
ac.eztrim(src_[0],[(O,O+1)], 		asrc_wkn, "Magia12c_cut.mka")
ac.eztrim(src_[0],[(7145,12276)], 	asrc_wkn, "Magia12d_cut.mka")
ac.eztrim(src_[0],[(O,O+1)], 		asrc_wkn, "Magia12e_cut.mka")
ac.eztrim(src_[0],[(12276,-10)], 	asrc_wkn, "Magia12f_cut.mka")

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
        clip_fu[0] + clip[:4986] + 
        clip_op_ed[4986:7144] + clip_op_ed[7143] + clip[7145:12276] +
        clip[12275] + clip[12276:31768] +
        clip_op_ed[31768:-10]
    )

src_ = [slicing(src, src_[1], src_fu) for src in src_]
src_fu = src_fu = src_fu[289:34217] + src_fu[34457:34577]
#src_amz = src_amz[:7146] + src_amz[7147:12279] + src_amz[12280:33930] + src_amz[34171:]

# Slicing comparison
#scomp = core.std.StackVertical([lvf.scomp(src_[0], src_fu, src_amz, make_diff=False, warn=False), 
#								lvf.scomp(src_[1], src_[2], src_[3], make_diff=False, warn=False)])

# Variable & Dither
src_ = [fvf.Depth(src, 16) for src in src_]
src_fu = fvf.Depth(src_fu, 16)
src = src_[0]

src_fade = kgf.hardsubmask_fades(src, src_fu, expand_n=12, highpass=2000)
src = fvf.rfs(src, core.std.MaskedMerge(src, src_fu, src_fade), 
    mappings="[1902 1985] [8202 8376] [9666 9709] [9752 9786]" +
    "[9971 10071] [11891 11925] [11927 11949] [16840 16921] [25556 25772]" + 
    "[31650 31769]")

#scomp2 = core.std.StackVertical([lvf.stack_compare(src, src_fu, src_amz, height=486, make_diff=False, warn=False),
#								lvf.stack_compare(src_sc, src_[2], src_[3], height=486, make_diff=False, warn=False)])

edstart, edend = 31770, 33927
ending_a = f"[{edstart+284} {edstart+452}]"
ending_b = f"[{edstart+1870} {edstart+1900}]"

den_a = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4)
den_b = vrdf.knlmcl(src, 1, 1, 0, 16)
den = fvf.rfs(den_a, den_b, mappings=ending_a + ending_b)

aa = taa.TAAmbk(den, aatype='Nnedi3', nsize=3, nns=3, down8=False)

adapt_m = core.adg.Mask(aa.std.PlaneStats(), 28)
db_a = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 4, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 17, 56, 48, 48, 20, 0, sample_mode=4)
db_c = dbs.f3kbilateral(aa, 17, 64, 64)
db_e = dbs.f3kbilateral(aa, 15, 48, 48)
db_f = dbs.f3kbilateral(aa, 22, 160, 160)
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = fvf.rfs(db, db_c, mappings="[25209 25532]")
db = fvf.rfs(db, db_e, mappings=ending_a)
db = fvf.rfs(db, db_f, mappings=ending_b)

grain_a = kgf.adaptive_grain(db, 0.25)
grain_b = core.grain.Add(db, 0.5, 0.5, constant=True)
grain_c = core.grain.Add(db, 0.5, constant=True)
grain = fvf.rfs(grain_a, grain_b, mappings=ending_a)
grain = fvf.rfs(grain, grain_c, mappings=ending_b)

endcard = fvf.rfs(grain, den, mappings="[33928 34046]")

final = fvf.Depth(endcard, 10)

final.set_output()