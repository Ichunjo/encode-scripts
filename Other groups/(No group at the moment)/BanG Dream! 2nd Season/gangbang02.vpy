import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from vsutil import get_y, split, join

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000
core.num_threads = 14

src = lvf.src(r"BRMM-10190BD/BDMV/STREAM/00001.m2ts")
#asrc = r"BRMM-10190BD/BDMV/STREAM/00001_T2_Audio - Japanese.w64"
#ac.eztrim(src, [(0,34048)], asrc, "02-cut.mka")

src = src[:34048]

opstart = 2158
edstart, edend = 31530, 33687

crop = core.std.CropRel(src, top=206, bottom=206)
fixedges = core.edgefixer.ContinuityFixer(crop, 0, [1,0,0], 0, [1,0,0])
ed_a = core.std.AddBorders(fixedges, top=206, bottom=206)

crop = core.std.CropRel(src, left=1078)
fixedges = core.edgefixer.ContinuityFixer(crop, [2,0,0], 0, 0, 0)
borders = core.std.AddBorders(fixedges, left=1078)
mask_borders_b = kgf.squaremask(src, 842, 1080, 1078, 0)
borders = core.std.MaskedMerge(src, borders, mask_borders_b)
ed_b = lvf.wipe_row(borders, width=6, height=1080, offset_x=1072)

crop = core.std.CropRel(src, right=1076)
fixedges = core.edgefixer.ContinuityFixer(crop, 0, 0, [1,0,0], 0)
borders = core.std.AddBorders(fixedges, right=1076)
mask_borders_c = kgf.squaremask(src, 844, 1080, 0, 0)
borders = core.std.MaskedMerge(src, borders, mask_borders_c)
ed_c = lvf.wipe_row(borders, width=6, height=1080, offset_x=844)

ed = fvf.rfs(src, ed_a, mappings=f"[{edstart} {edstart+378}] [{edstart+1138} {edstart+1159}]")
ed = fvf.rfs(ed, ed_b, mappings=f"[{edstart+393} {edstart+594}] [{edstart+812} {edstart+1025}]")
ed = fvf.rfs(ed, ed_c, mappings=f"[{edstart+596} {edstart+810}] [{edstart+1027} {edstart+1137}]")

ed = core.std.Expr(ed, ['x 16 <= x 0 = x ?', '', ''])
ed = core.std.Expr(ed, ['x 1 = x 0 = x ?', '', ''])

src = fvf.rfs(src, ed, mappings=f"[{edstart} {edend}]")

src = fvf.Depth(src, 16)

luma = get_y(src)

descaled = kgf.inverse_scale(luma, 1433, 806, kernel='bicubic', b=0, c=1/2, mask_detail=True, descale_mask_zones='[2158 4316] [25626 25694] [32691 33687] [33881 34003]')
descaled16 = fvf.Depth(descaled, 16)

denoise_y = mvf.BM3D(descaled, sigma=2.1, radius1=1, depth=16)
contra = hvf.ContraSharpening(denoise_y, descaled16, 13)
contra = fvf.rfs(contra, descaled16, mappings=f"[{opstart+1160} {opstart+1391}]")

deband_mask = kgf.retinex_edgemask(contra).std.Inflate().std.Deflate()
deband_mask = core.std.Expr(deband_mask, 'x 30 < 0 x 3 * ?')
db_y, db_c = vrdf.F3kdbSep(contra, src, 16, 46, 40, 0, 0, mask=deband_mask)
db_y_b, db_c_b = vrdf.F3kdbSep(contra, src, 17, 56, 56, 0, 0)
db_y_c, db_c_c = vrdf.F3kdbSep(db_y, db_c, 8, 64, 64, 0, 0)
db_y_d, db_c_d = vrdf.F3kdbSep(contra, src, 7, 90, 60, 0, 0)

db_y = fvf.rfs(db_y, db_y_b, mappings=f"[{opstart+555} {opstart+670}]")
db_c = fvf.rfs(db_c, db_c_b, mappings=f"[{opstart+555} {opstart+670}]")
db_y = fvf.rfs(db_y, db_y_c, mappings=f"[{opstart+1496} {opstart+1502}]")
db_c = fvf.rfs(db_c, db_c_c, mappings=f"[{opstart+1496} {opstart+1502}]")
db_y = fvf.rfs(db_y, db_y_d, mappings=f"[{opstart+126} {opstart+554}]")
db_c = fvf.rfs(db_c, db_c_d, mappings=f"[{opstart+126} {opstart+554}]")

aa = lvf.upscaled_sraa(db_y, 1.75, 13, 1080, True)
repmask_aa = kgf.squaremask(aa, width=1920, height=1076, offset_x=0, offset_y=2)
aa = core.std.MaskedMerge(luma, aa, repmask_aa)

grain_1 = core.f3kdb.Deband(aa, grainy=24, output_depth=16, preset='depth')
grain_2 = core.f3kdb.Deband(grain_1, grainy=80, dynamic_grain=True, output_depth=16, preset='depth')
grain = fvf.rfs(grain_1, grain_2, mappings=f"[{opstart+126} {opstart+554}]")
grain_a = kgf.adaptive_grain(grain, 0.2)
grain_b = kgf.adaptive_grain(grain, 0.45, luma_scaling=8, static=False)
grain_c = kgf.adaptive_grain(grain, 0.3, luma_scaling=16, static=False)
grain_d = kgf.adaptive_grain(grain, 0.3, luma_scaling=6)
grain = fvf.rfs(grain_a, grain_b, mappings="[582 676] [26817 29180]")
grain = fvf.rfs(grain, grain_c, mappings=f"[{opstart+555} {opstart+670}]")
grain = fvf.rfs(grain, grain_d, mappings=f"[{opstart+1496} {opstart+1502}]")

planes = split(db_c)

merged = join([grain, planes[1], planes[2]])

rep_edmask_a = lvf.wipe_row(src, width=1920, height=206+2, offset_x=0, offset_y=0, width2=1920, height2=206+6, offset_x2=0, offset_y2=874-6, show_mask=1)
rep_edmask_a = hvf.mt_inflate_multi(rep_edmask_a, radius=4)
rep_edmask_b = fvf.Depth(mask_borders_b, 16)
rep_edmask_c = fvf.Depth(mask_borders_c, 16)

ed16 = fvf.Depth(ed, 16)

rep_ed = fvf.rfs(merged, core.std.MaskedMerge(merged, ed16, rep_edmask_a), mappings=f"[{edstart} {edstart+378}] [{edstart+1138} {edstart+1159}]")
rep_ed = fvf.rfs(rep_ed, core.std.MaskedMerge(merged, ed16, rep_edmask_b), mappings=f"[{edstart+393} {edstart+594}] [{edstart+812} {edstart+1025}]")
rep_ed = fvf.rfs(rep_ed, core.std.MaskedMerge(merged, ed16, rep_edmask_c), mappings=f"[{edstart+596} {edstart+810}] [{edstart+1027} {edstart+1137}]")

final = fvf.Depth(rep_ed, 10)

final.set_output()