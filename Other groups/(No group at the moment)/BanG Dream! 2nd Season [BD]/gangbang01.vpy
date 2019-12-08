import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
import vsTAAmbk as taa
from vsutil import get_y, split, join

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000
core.num_threads = 14

src = lvf.src(r"BRMM-10190BD/BDMV/STREAM/00000.m2ts")
asrc = r"BRMM-10190BD/BDMV/STREAM/00000_T2_Audio - Japanese.w64"
#ac.eztrim(src, [(0,34047)], asrc, "01-cut.mka")
#ac.eztrim(src, [(2997,5154)], asrc, "01_op-cut.mka")

src = src[:34047]
src = fvf.Depth(src, 16)

opstart = 2997

luma = get_y(src)

descaled = kgf.inverse_scale(luma, 1433, 806, kernel='bicubic', b=0, c=1/2, mask_detail=True, descale_mask_zones='[2997 5274] [32372 34046]')

denoise_y = mvf.BM3D(descaled, sigma=2.1, radius1=1, depth=16)
contra = hvf.ContraSharpening(denoise_y, fvf.Depth(descaled, 16), 13)

deband_mask = kgf.retinex_edgemask(contra).std.Inflate().std.Deflate()
deband_mask = core.std.Expr(deband_mask, 'x 30 < 0 x 3 * ?')
db_y, db_c = vrdf.F3kdbSep(contra, src, 16, 46, 40, 0, 0, mask=deband_mask)
db_y_b = fvf.gf3(contra, smode=2, radius=8, thr=2)
db_c_b = fvf.gf3(src, smode=2, radius=8, thr=2, planes=[1, 2])
db_y_b, db_c_b = vrdf.F3kdbSep(db_y_b, db_c_b, 17, 30, 30, 0, 0)
db_y_c, db_c_c = vrdf.F3kdbSep(contra, src, 12, 48, 42, 0, 0, mask=deband_mask)
db_y = fvf.rfs(db_y, db_y_b, mappings=f"[{opstart+855} {opstart+865}]")
db_c = fvf.rfs(db_c, db_c_b, mappings=f"[{opstart+855} {opstart+865}]")
db_y = fvf.rfs(db_y, db_y_c, mappings="[33937 33994]")
db_c = fvf.rfs(db_c, db_c_c, mappings="[33937 33994}]")

aa = lvf.upscaled_sraa(db_y, 1.75, 13, 1080, True)
repmask_aa = kgf.squaremask(aa, width=1920, height=1076, offset_x=0, offset_y=2)
aa = core.std.MaskedMerge(luma, aa, repmask_aa)
aa_b = taa.TAAmbk(aa, aatype='Nnedi3', mtype=1)
aa = fvf.rfs(aa, aa_b, mappings="[13303 13359]")

grain_1 = core.f3kdb.Deband(aa, grainy=24, output_depth=16, preset='depth')
grain_2 = core.f3kdb.Deband(aa, grainy=64, output_depth=16, preset='depth')
grain = fvf.rfs(grain_1, grain_2, mappings=f"[{opstart+855} {opstart+865}]")
grain_a = kgf.adaptive_grain(grain, 0.2)
grain_b = kgf.adaptive_grain(grain, 0.8, luma_scaling=4)
grain_roselia_a = kgf.adaptive_grain(grain, 0.45, luma_scaling=8, static=False)
grain_roselia_b = kgf.adaptive_grain(grain, 0.3, static=False)
grain_yukina = kgf.adaptive_grain(grain, 1.2, luma_scaling=2, static=False)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+855} {opstart+865}]")
grain = fvf.rfs(grain, grain_roselia_a, mappings="[24956 26386] [33744 34046]")
grain = fvf.rfs(grain, grain_roselia_b, mappings="[26387 26832] [26860 27459] [33687 33743]")
grain = fvf.rfs(grain, grain_yukina, mappings="[25633 25644] [26833 26859]")

planes = split(db_c)

merged = join([grain, planes[1], planes[2]])

c_mask = vrdf.DRM(src, 806, 'bicubic', b=0, c=1/2, sw=2, sh=2).std.Inflate()
credit = fvf.rfs(merged, core.std.MaskedMerge(merged, src, c_mask), mappings='[2997 5274] [32372 34046]')

final = fvf.Depth(credit, 10)

final.set_output()