import vapoursynth as vs
core = vs.get_core()
import nnedi3_resample as nresize
import muvsfunc as muvf
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import adjust
import vardefunc as vrdf

core.max_cache_size = 10000

src = lvf.src(r'The Demon Girl Next Door E03 [1080p][E-AC3][JapDub][EngSub][Web-DL].mkv')[24:34670]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 1248

dblk = fvf.AutoDeblock(src)
dblk = fvf.rfs(src, dblk, mappings=f"[315 1247] [{opstart+1149} {opstart+1218}] [3525 5255] [5379 5420] [5547 6076] [6489 8345] [8405 8934] [9104 9546] [9678 10664] [10722 10812]")

denoise_a = vrdf.KNLMeansCL(dblk, 0.5, 0.3, 1)
denoise_b = vrdf.KNLMeansCL(dblk, 0.8, 0.6, 1)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f"[{opstart+1053} {opstart+1070}] [315 1247] [3525 5255]")

sharpen = hvf.ContraSharpening(denoise, src)

deband_a = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(sharpen, range=18, y=46, cb=38, cr=38, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=f"[315 1247] [{opstart+1149} {opstart+1218}] [3525 5255] [5379 5420] [5547 6076] [6489 8345] [8405 8934] [9104 9546] [9678 10664] [10722 10812]")

aa_a = taa.TAAmbk(deband, aatype='Nnedi3', cycle=1, opencl=True, opencl_device=2)
aa_b = taa.TAAmbk(deband, aatype='Eedi3SangNom', sharp=120, opencl=True, opencl_device=2)
aa = fvf.rfs(aa_a, aa_b, mappings="[24573 24662]")

dimming_a = adjust.Tweak(aa, sat=1.20, cont=1.20)
dimming = fvf.rfs(aa, dimming_a, mappings="[11336 11425] [30877 31057]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()