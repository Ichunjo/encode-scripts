import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import fag3kdb as f3kdb
import havsfunc as hvf
import lvsfunc as lvf
import adjust
import vardefunc as vrdf
import havsfunc as hvf

core.max_cache_size = 12500

src = lvf.src(r'The Demon Girl Next Door E07 [1080p][E-AC3][JapDub][EngSub][Web-DL].mkv')[24:34669]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)
src = src+src[34643:34644]

opstart = 2781

denoise = vrdf.KNLMeansCL(src, 0.4, 0.2, 1)

sharpen = hvf.ContraSharpening(denoise, src)

deband = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)

aa_a = taa.TAAmbk(deband, aatype='Nnedi3', opencl=True, opencl_device=2)

grain_a = kgf.adaptive_grain(aa_a, 0.3)
grain_b = kgf.adaptive_grain(aa_a, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()