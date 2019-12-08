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

src = lvf.src(r'The Demon Girl Next Door E05 [1080p][E-AC3][JapDub][EngSub][Web-DL].mkv')[24:34670]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 888

dblk = fvf.AutoDeblock(src)
dblk = fvf.rfs(src, dblk, mappings=f"[{opstart+1149} {opstart+1218}] [25436 26375] [34352 34384]")

denoise_a = vrdf.KNLMeansCL(dblk, 0.5, 0.3, 1)
denoise_b = vrdf.KNLMeansCL(dblk, 0.8, 0.6, 1)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f"[{opstart+1053} {opstart+1070}] [25436 26375] [34352 34384]")

sharpen = hvf.ContraSharpening(denoise, src)

deband_a = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(sharpen, range=18, y=46, cb=38, cr=38, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=f"[{opstart+1149} {opstart+1218}] [34352 34384]")

aa_a = taa.TAAmbk(deband, aatype='Nnedi3', cycle=1, opencl=True, opencl_device=2)
aa_b = taa.TAAmbk(deband, aatype='Eedi3SangNom', strength=128, cycle=4, opencl=True)
aa_c = taa.TAAmbk(deband, aatype='Eedi3SangNom', strength=128, cycle=2)
aa_d = taa.TAAmbk(deband, aatype='Eedi3', opencl=True, opencl_device=2)
aa = fvf.rfs(aa_a, aa_b, mappings="21204")
aa = fvf.rfs(aa, aa_c, mappings="21205")
aa = fvf.rfs(aa, aa_d, mappings="[24451 24507]")

dimming_a = adjust.Tweak(aa, sat=1.25, cont=1.25)
dimming_b = adjust.Tweak(aa, sat=1.65, cont=1.65)
dimming = fvf.rfs(aa, dimming_a, mappings="[21050 21200]")
dimming = fvf.rfs(dimming, dimming_b, mappings="[25436 26375]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()