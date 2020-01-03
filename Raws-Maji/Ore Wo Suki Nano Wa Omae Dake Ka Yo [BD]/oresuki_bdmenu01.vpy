import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf

core = vs.core
core.max_cache_size = 26000

src = lvf.src(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00018.m2ts")

src = fvf.Depth(src, 32)

den = vrdf.KNLMCL(src, 1.2, 1.2, 0, 16)

aa = lvf.upscaled_sraa(den, rep=6, sharp_downscale=False)
aa = vrdf.to444(aa)

db = core.neo_f3kdb.Deband(aa, 17, 30, 30, 30, 12, 0, sample_mode=4)

final = fvf.Depth(db, 10)

final.set_output()