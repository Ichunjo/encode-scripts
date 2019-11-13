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

src = lvf.src(r'joshimuda01.m2ts')[:34046]
src = fvf.Depth(src, 16)

opstart, opend = 1582, 3740
eyestart, eyeend = 16232, 16351
edstart, edend = 29371, 31527
last = src.num_frames-1

op_filter = f'[{opstart+337} {opstart+358}] [{opstart+388} {opstart+406}] [{opstart+521} {opstart+542}] [{opstart+573} {opstart+587}] [{opstart+696} {opstart+715}] [{opstart+757} {opstart+768}] [{opstart+793} {opstart+813}] [{opstart+857} {opstart+877}] [{opstart+894} {opstart+1575}] [{opstart+1758} {opstart+2067}]'

luma = core.std.ShufflePlanes(src, 0, vs.GRAY)

descaled = core.descale.Debicubic(fvf.Depth(luma, 32), get_w(864), 864, 0, 1/2)
descaled_eye = core.descale.Debicubic(fvf.Depth(luma, 32), get_w(878), 878, 0, 1/2)
descaled = fvf.Depth(descaled, 16)

denoise_l = hvf.SMDegrain(descaled, tr=2, thSAD=280, contrasharp=True)
denoise_l = fvf.rfs(denoise_l, descaled, mappings=op_filter)

scaled = nnedi3_rpow2CL(denoise_l, device=0).resize.Spline36(1920, 1080)
scaled_eye = nnedi3_rpow2CL(descaled_eye, device=0).resize.Spline36(1920, 1080, format=vs.GRAY16)
scaled = fvf.rfs(scaled, scaled_eye, mappings=f'[{eyestart} {eyeend}]')
scaled = fvf.rfs(scaled, luma, mappings='[29268 29370]')

c_mask = hvf.mt_expand_multi(vrdf.DRM(luma, 864, 'bicubic', b=0, c=1/2, mthr=110), mode='ellipse', sw=2, sh=2).std.Inflate()
credit = fvf.rfs(scaled, core.std.MaskedMerge(scaled, luma, c_mask), mappings=f'[{opstart} {opend+147}] [{edstart} {edend}]')

merged = core.std.ShufflePlanes([credit, src], [0, 1, 2], vs.YUV)

aa_a = taa.TAAmbk(merged, aatype='Eedi3', mtype=2)
aa_b = taa.TAAmbk(merged, aatype='Eedi3', mtype=2, cycle=2)
aa_c = taa.TAAmbk(aa_b, aatype='Eedi3SangNom', mtype=2, repair=2)
aa_d = taa.TAAmbk(merged, aatype='Nnedi3', mtype=1)
aa = fvf.rfs(merged, aa_a, mappings=f'[{opstart+382} {opstart+447}] [17433 17512]')
aa = fvf.rfs(aa, aa_b, mappings='[32854 32973] [33582 33689]')
aa = fvf.rfs(aa, aa_c, mappings='[8137 8388] [11303 11434]')
aa = fvf.rfs(aa, aa_d, mappings='[29268 29370]')

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