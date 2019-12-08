import vapoursynth as vs
core = vs.get_core()
import muvsfunc as muvf
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_w, get_y
core.max_cache_size = 26000

src = lvf.src(r'joshimuda03.m2ts')[:34048]
src = fvf.Depth(src, 16)

opstart, opend = 2158, 4316
eyestart, eyeend = 19949, 20068
edstart, edend = 30283, 32441

op_filter = f'[{opstart+337} {opstart+358}] [{opstart+388} {opstart+406}] [{opstart+521} {opstart+542}] [{opstart+573} {opstart+587}] [{opstart+696} {opstart+715}] [{opstart+757} {opstart+768}] [{opstart+793} {opstart+813}] [{opstart+857} {opstart+877}] [{opstart+894} {opstart+1575}] [{opstart+1758} {opstart+2067}]'

luma = core.std.ShufflePlanes(src, 0, vs.GRAY)

descaled = core.descale.Debicubic(fvf.Depth(luma, 32), get_w(864), 864, 0, 1/2)
descaled = fvf.Depth(descaled, 16)

denoise_l = hvf.SMDegrain(descaled, tr=2, thSAD=280, contrasharp=True)
denoise_l = fvf.rfs(denoise_l, descaled, mappings=op_filter)

scaled = nnedi3_rpow2CL(denoise_l, device=0).resize.Spline36(1920, 1080)

c_mask = hvf.mt_expand_multi(vrdf.DRM(luma, 864, 'bicubic', b=0, c=1/2, mthr=110), mode='ellipse', sw=2, sh=2).std.Inflate()
credit = fvf.rfs(scaled, core.std.MaskedMerge(scaled, luma, c_mask), mappings=f'[{opstart} 4423] [{edstart} {edend}]')

merged = core.std.ShufflePlanes([credit, src], [0, 1, 2], vs.YUV)

aa_a = taa.TAAmbk(merged, aatype='Eedi3', mtype=2)
aa_b = lvf.upscaled_sraa(get_y(merged), 1.5)
aa_b = core.std.ShufflePlanes([aa_b, merged], [0, 1, 2], vs.YUV)
aa_b = core.warp.AWarpSharp2(aa_b, thresh=32, blur=3, type=0, depth=16)
aa = fvf.rfs(merged, aa_a, mappings=f'[{opstart+382} {opstart+447}]')
aa = fvf.rfs(aa, aa_b, mappings='[88 1174] [1207 1822]')

deband_a = core.f3kdb.Deband(aa, range=17, y=36, cb=26, cr=26, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=14, y=20, cb=12, cr=12, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=op_filter)
line = kgf.retinex_edgemask(aa)
deband = core.std.MaskedMerge(deband, aa, line)

grain_a = kgf.adaptive_grain(deband, 0.2, luma_scaling=6)
grain_b = kgf.adaptive_grain(grain_a, 0.8, static=False, luma_scaling=200)
grain = fvf.rfs(grain_b, deband, mappings=op_filter)

final = fvf.Depth(grain, 10)
final.set_output()