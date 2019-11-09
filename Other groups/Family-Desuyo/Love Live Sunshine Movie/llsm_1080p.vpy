import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
from nnedi3_rpow2CL import nnedi3_rpow2CL

core.max_cache_size = 16000

src = lvf.src(r'00004.m2ts')[384:144192]
nced = lvf.src(r'00024.m2ts')[24:11952]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
nced = core.std.AssumeFPS(nced, fpsnum=24000,fpsden=1001)

src = src[0:131880]+nced

fixedges_a = core.edgefixer.ContinuityFixer(src, [2,1,1], [2,1,1], [2,1,1], [2,1,1])
fixedges_b = core.edgefixer.ContinuityFixer(src, [2,1,1], [2,1,1], [6,3,3], [2,1,1], [2,1,1])
fixedges = fvf.rfs(fixedges_a, fixedges_b, mappings="[104359 104361]")

fixedges = fvf.Depth(fixedges, 16)

Y, U, V = kgf.split(fixedges)
descaled_y = kgf.inverse_scale(Y, height=720, kernel='bilinear', mask_detail=True, descale_mask_zones="[0 887] [9531 10148]")
upscaled_y = nnedi3_rpow2CL(descaled_y, device=2).resize.Bicubic(1920, 1080, format=vs.GRAY16, filter_param_a=0, filter_param_b=0)
scaled = kgf.join([upscaled_y, U, V])

denoise = vrdf.KNLMeansCL(scaled, 0.75, 0.35, 1)

deband_a = core.f3kdb.Deband(denoise, range=16, y=36, cb=24, cr=24, grainy=10, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(denoise, range=20, y=48, cb=36, cr=36, grainy=20, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings="[39581 39643] [49491 49540] [76644 76676] [105779 105988] [108199 108276] [108368 108544] [123942 124037] [132295 132324]")

aa_a = taa.TAAmbk(deband, aatype='Nnedi3SangNom', sharp=120, repair=2, opencl=True, opencl_device=2)
aa_b = taa.TAAmbk(deband, aatype='Eedi3', cycle=2, sharp=120, opencl=True, opencl_device=2)
aa = fvf.rfs(aa_a, aa_b, mappings="[79523 79576]")

grain = kgf.adaptive_grain(aa, 0.4, static=True)
grain = kgf.adaptive_grain(grain, 0.6, static=False)

final = fvf.Depth(grain, 10)
final.set_output()