import vapoursynth as vs
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vsTAAmbk as taa

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 14

src = lvf.src(r'BDMV/STREAM/00004.m2ts')

src = core.std.FreezeFrames(src, 24, 1092, 30)

src = fvf.Depth(src, 16)

dering = hvf.HQDeringmod(src, mrad=2, msmooth=1, mthr=160, drrep=13, thr=64, elast=1, show=False)

aa = taa.TAAmbk(dering, aatype='Nnedi3', mtype=1, nns=4, nsize=5, down8=False)

final = fvf.Depth(aa, 10)

final.set_output()