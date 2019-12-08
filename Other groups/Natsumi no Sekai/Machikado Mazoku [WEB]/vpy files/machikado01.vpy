import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import adjust
import vardefunc as vrdf
import havsfunc as hvf

core.max_cache_size = 10000

src = lvf.src(r'The Demon Girl Next Door E01 [1080p][E-AC3][JapDub][EngSub][Web-DL].mkv')[24:34669]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 2158

dblk = fvf.AutoDeblock(src)
dblk = fvf.rfs(src, dblk, mappings=f"[{opstart+1149} {opstart+1218}] [0 1480] [12168 12667] [34606 34644]")

denoise_a = vrdf.KNLMeansCL(dblk, 0.5, 0.3, 1)
denoise_b = vrdf.KNLMeansCL(dblk, 0.8, 0.6, 1)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f"[{opstart+1053} {opstart+1070}] [0 1480]")

sharpen = hvf.ContraSharpening(denoise, src)

deband_a = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(sharpen, range=18, y=46, cb=38, cr=38, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=f"[{opstart+1149} {opstart+1218}] [0 1480] [34606 34644] ")

aa_a = taa.TAAmbk(deband, aatype='Nnedi3', opencl=True, opencl_device=2)
aa_b = taa.TAAmbk(deband, aatype='Eedi3', opencl=True, opencl_device=2)
aa_c = taa.TAAmbk(deband, aatype='Eedi3SangNom', cycle=2, opencl=True, opencl_device=2)
aa = fvf.rfs(aa_a, aa_b, mappings="[12168 12667] [34612 34644]")
aa = fvf.rfs(aa, aa_c, mappings="[34606 34611]")

dimming_a = adjust.Tweak(aa, sat=1.3, cont=1.3)
dimming_b = adjust.Tweak(aa, sat=1.45, cont=1.45)
dimming_c = adjust.Tweak(aa, sat=1.5, cont=1.5)
dimming = fvf.rfs(aa, dimming_a, mappings="[11604 11732]")
dimming = fvf.rfs(dimming, dimming_b, mappings="[12168 12667]")
dimming = fvf.rfs(dimming, dimming_c, mappings="[12413 12437]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()