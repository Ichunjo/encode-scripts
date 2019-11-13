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

core.max_cache_size = 64000

src = lvf.src(r'The Demon Girl Next Door E12 [1080p][AAC][JapDub][EngSub][Web-DL].mkv')[24:34309]
src = src+src[-1]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 5011

denoise = mvf.BM3D(src, [1.7, 0.85])

sharpen = hvf.ContraSharpening(denoise, src)

deband_a = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(sharpen, range=18, y=48, cb=36, cr=36, grainy=10, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings="[11819 12198]")

aa = taa.TAAmbk(deband, aatype='Nnedi3')

dimming_a = adjust.Tweak(aa, sat=1.8, cont=1.8)
dimming = fvf.rfs(aa, dimming_a, mappings="[11819 12198]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)

final.set_output()