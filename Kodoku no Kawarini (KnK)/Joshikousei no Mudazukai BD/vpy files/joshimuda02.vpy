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
from vsutil import get_w
core.max_cache_size = 26000

src = lvf.src(r'joshimuda02.m2ts')[:34046]
src = fvf.Depth(src, 16)

opstart, opend = 1894, 4052
eyestart, eyeend = 17191, 17310
edstart, edend = 27095, 29251

op_filter = f'[{opstart+337} {opstart+358}] [{opstart+388} {opstart+406}] [{opstart+521} {opstart+542}] [{opstart+573} {opstart+587}] [{opstart+696} {opstart+715}] [{opstart+757} {opstart+768}] [{opstart+793} {opstart+813}] [{opstart+857} {opstart+877}] [{opstart+894} {opstart+1575}] [{opstart+1758} {opstart+2067}]'

luma = core.std.ShufflePlanes(src, 0, vs.GRAY)

descaled = core.descale.Debicubic(fvf.Depth(luma, 32), get_w(864), 864, 0, 1/2)
descaled = fvf.Depth(descaled, 16)

denoise_l = hvf.SMDegrain(descaled, tr=2, thSAD=280, contrasharp=True)
denoise_l = fvf.rfs(denoise_l, descaled, mappings=op_filter)

scaled = nnedi3_rpow2CL(denoise_l, device=0).resize.Spline36(1920, 1080)

c_mask = hvf.mt_expand_multi(vrdf.DRM(luma, 864, 'bicubic', b=0, c=1/2, mthr=110), mode='ellipse', sw=2, sh=2).std.Inflate()
credit = fvf.rfs(scaled, core.std.MaskedMerge(scaled, luma, c_mask), mappings=f'[{opstart} {opend+107}] [{edstart} {edend}]')

merged = core.std.ShufflePlanes([credit, src], [0, 1, 2], vs.YUV)

aa_a = taa.TAAmbk(merged, aatype='Eedi3', mtype=2)
aa = fvf.rfs(merged, aa_a, mappings=f'[{opstart+382} {opstart+447}] [17311 19701] [26981 27094]')

deband_a = core.f3kdb.Deband(aa, range=17, y=36, cb=26, cr=26, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=14, y=20, cb=12, cr=12, grainy=0, grainc=0, output_depth=16)
deband_c = core.f3kdb.Deband(aa, range=18, y=46, cb=38, cr=38, grainy=18, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=op_filter)
deband = fvf.rfs(deband, deband_c, mappings='[19765 19836] [19885 20004]')
line = kgf.retinex_edgemask(aa)
deband = core.std.MaskedMerge(deband, aa, line)

grain_a = kgf.adaptive_grain(deband, 0.2, luma_scaling=6)
grain_b = kgf.adaptive_grain(grain_a, 0.8, static=False, luma_scaling=200)
grain = fvf.rfs(grain_b, deband, mappings=op_filter)


final = fvf.Depth(grain, 10)
final.set_output()