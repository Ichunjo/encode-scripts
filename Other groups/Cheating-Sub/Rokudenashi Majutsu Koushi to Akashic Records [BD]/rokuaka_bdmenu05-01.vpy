import vapoursynth as vs
import fvsfunc as fvf
import lvsfunc as lvf
core = vs.core

src = lvf.src(r'[BDMV] Rokudenashi Majutsu Koushi to Akashic Records/Vol 5/BDMV/STREAM/00000.m2ts')

src = src[2453]
src = core.std.AssumeFPS(src, fpsnum=1000, fpsden=213880)

src = fvf.Depth(src, 16)

aa = lvf.upscaled_sraa(src, 2)

deband = core.f3kdb.Deband(aa, range=16, y=24, cb=24, cr=24, grainy=16, grainc=0, output_depth=16)

final = fvf.Depth(deband, 10)

final.set_output()