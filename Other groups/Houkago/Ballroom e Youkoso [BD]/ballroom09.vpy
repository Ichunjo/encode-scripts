import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrf
import mvsfunc as mvf
import modfunc as mdf
from nnedi3_rpow2CL import nnedi3_rpow2CL

core = vs.core
core.max_cache_size = 8192

src = lvf.src(r'[BDMV][180124][Ballroom e Youkoso][Vol.03]\BALLROOM3\DISK1\BDMV\STREAM\00000.m2ts')

src = src[:-24]

src = fvf.Depth(src, 32)
y = lvf.get_y(src)

den_uv = core.knlm.KNLMeansCL(src, a=2, h=0.6, d=3, device_type='gpu', channels='UV')

desc_y = core.descale.Debicubic(y, lvf.get_w(864), 864, 0.2620, 0.3690, cache_size=5)

den_y = mvf.BM3D(desc_y, 2, depth=16)

upsc_y = nnedi3_rpow2CL(den_y, 2, 1920, 1080, kernel='spline64', nsize=1, nns=4, qual=2)
scaled = core.std.ShufflePlanes([upsc_y, fvf.Depth(den_uv, 16)], [0, 1, 2], vs.YUV)

credit_m = vrf.drm(y, 864, 'Spline36', 0, 0, 120, 'ellipse', 4, 4).std.Inflate()
scaled = core.std.MaskedMerge(scaled, fvf.Depth(src, 16), fvf.Depth(credit_m, 16), 0)

db = core.neo_f3kdb.Deband(fvf.Depth(scaled, 16), 17, 36, 36, 36, 32, 0, sample_mode=4)
grain = mdf.adptvgrnMod_mod(db, 0.3, size=1.25, sharp=40, luma_scaling=8)


final = core.fmtc.bitdepth(grain, bits=10, dmode=3)

final.set_output()