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

src = lvf.src(r'The Demon Girl Next Door E06 [1080p][E-AC3][JapDub][EngSub][Web-DL].mkv')[24:34668]
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = fvf.Depth(src, 16)
src = src+src[34643:34644]+src[34643:34644]

opstart = 982

dblk = fvf.AutoDeblock(src)
dblk = fvf.rfs(src, dblk, mappings="[8889 9412]")

denoise = vrdf.KNLMeansCL(dblk, 0.4, 0.2, 1)

sharpen = hvf.ContraSharpening(denoise, src)

deband = core.f3kdb.Deband(sharpen, range=16, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)

aa_a = taa.TAAmbk(deband, aatype='Nnedi3')
aa_b = taa.TAAmbk(deband, aatype='Nnedi3', thin=15, sharp=60)
aa_c = taa.TAAmbk(deband, aatype='Nnedi3', thin=10, sharp=40)
aa_d = taa.TAAmbk(deband, aatype='Eedi3')
aa = fvf.rfs(aa_a, aa_b, mappings="[6199 6321]")
aa = fvf.rfs(aa_a, aa_c, mappings="6322")
aa = fvf.rfs(aa_a, aa_d, mappings="[8889 9412]")

dimming_a = adjust.Tweak(aa, sat=1.25, cont=1.25)
dimming_b = adjust.Tweak(aa, sat=1.8, cont=1.8)
dimming = fvf.rfs(aa, dimming_a, mappings="[6981 7092]")
dimming = fvf.rfs(dimming, dimming_b, mappings="[8889 9412]")

grain_a = kgf.adaptive_grain(dimming, 0.3)
grain_b = kgf.adaptive_grain(dimming, 0.8)
grain = fvf.rfs(grain_a, grain_b, mappings=f"[{opstart+1053} {opstart+1070}]")

final = fvf.Depth(grain, 10)
final.set_output()