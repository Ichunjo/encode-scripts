import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf

core = vs.get_core()
core.max_cache_size = 32000
#core.num_threads = 14

src = lvf.src(r'Vol.06 Fin/BDMV/STREAM/00002.m2ts')
src = fvf.Depth(src, 16)

freeze = core.std.FreezeFrames(src, 261, 3215, 2639)

aa = lvf.upscaled_sraa(freeze, 2, 24, sharp_downscale=True)

final = fvf.Depth(aa, 10)

final.set_output()