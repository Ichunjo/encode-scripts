import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vsTAAmbk as taa
import modfunc as mdf
import acsuite

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

ac = acsuite.AC()

src = lvf.src(r"BDROM Vol 4/BDMV/STREAM/00001.m2ts")

den = mdf.hybriddenoise_mod(src, knl=0.55, sigma=2.2)

aa = taa.TAAmbk(den, aatype='Nnedi3', opencl=True)

db = core.neo_f3kdb.Deband(aa, 16, 30, 30, 30, 24, 0, sample_mode=4)

grain = kgf.adaptive_grain(db, 0.3, luma_scaling=8)

final = fvf.Depth(grain, 10)

final.set_output()