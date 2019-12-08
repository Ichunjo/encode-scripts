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

src = lvf.src(r'The Demon Girl Next Door E10 [1080p][AAC][JapDub][EngSub][Web-DL].mkv')[12:34657]
src = src+src[-1]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 4651

denoise = mvf.BM3D(src, [1.7, 0.85])

sharpen = hvf.ContraSharpening(denoise, src)

deband_a = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(sharpen, range=18, y=48, cb=36, cr=36, grainy=10, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings="[566 1089] [2824 3595] [9469 9528]")

aa_a = taa.TAAmbk(deband, aatype='Nnedi3')
aa_b = taa.TAAmbk(deband, aatype='Eedi3SangNom', cycle=2, repair=2, sharp=80)
aa = fvf.rfs(aa_a, aa_b, mappings="[30237 30334]")

dimming_a = adjust.Tweak(aa, sat=1.35, cont=1.35)
dimming_b = adjust.Tweak(aa, sat=1.8, cont=1.8)
dimming_c = adjust.Tweak(aa, sat=1.45, cont=1.45)
dimming = fvf.rfs(aa, dimming_a, mappings="[323 411] [30237 30334]")
dimming = fvf.rfs(dimming, dimming_b, mappings="[566 1089]")
dimming = fvf.rfs(dimming, dimming_c, mappings="[2824 3378]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()