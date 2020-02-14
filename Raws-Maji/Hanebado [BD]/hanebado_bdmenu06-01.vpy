import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import modfunc as mdf
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

src = lvf.src(r'[BDMV][190213][Hanebado!][Vol.6]/BDMV/STREAM/00004.m2ts')

vrdf.generate_keyframes(src, 'hanebado_bdmenu06-01_keyframes.log')

src16 = fvf.Depth(src, 16)

den_pre = CoolDegrain(src16, tr=1, thsad=72, thsadc=48, bits=16, blksize=8, overlap=4)
den_a = mdf.hybriddenoise_mod(den_pre, knl=0.6, sigma=2.5, radius1=1)

edge_m = kgf.retinex_edgemask(den_pre, 0.1).std.Binarize(12000)
den = core.std.MaskedMerge(den_a, den_pre, edge_m)

aa = lvf.upscaled_sraa(den, rep=6, sharp_downscale=False)

db = core.neo_f3kdb.Deband(aa, 18, 48, 48, 48, 0, 0, sample_mode=4)

freeze = core.std.FreezeFrames(db, 134, 2111, 1412)

final = fvf.Depth(freeze, 10)

final.set_output()