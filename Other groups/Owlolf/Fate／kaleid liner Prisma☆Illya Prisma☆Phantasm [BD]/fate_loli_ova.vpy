import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import modfunc as mdf
import mvsfunc as mvf
import debandshit as dbs
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 16384

src = lvf.src(r'[BDMV][191127] Fate／kaleid liner Prisma☆Illya Prisma☆Phantasm\PRISMAPHANTASM_SP\BDMV\STREAM\00001.m2ts')

src = src[48:-24]

src = core.std.FreezeFrames(src, 3485, 3485, 3484)

ef = lvf.edgefixer(src, [1,0,0])

opstart, opend = 2789, 4946

src = fvf.Depth(ef, 32)

planes = lvf.split(src)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=0.25) for plane in planes[1:]]
shift = lvf.join(planes)

#den = core.knlm.KNLMeansCL(shift, a=2, h=0.4, d=3, device_type='gpu', channels='UV')
den_a = mvf.BM3D(shift, [1.5, 1], radius1=1)
den_b = mvf.BM3D(shift, [2.5, 1], radius1=1)
den = fvf.rfs(den_a, den_b, mappings='[82330 82412] [83236 83318] [84097 84767]')

desc_str = 0.85
y = lvf.get_y(den)
desc = core.descale.Debilinear(y, 1280, 720, cache_size=5)
down = core.resize.Spline36(y, 1280, 720)
desc = core.std.Expr([desc, down], f'x {desc_str} * y 1 {desc_str} - * +')


scaled = lvf.sraa(desc, 2, None, 1080, None, False)
line_m = TAAmbk(scaled.resize.Point(format=vs.GRAY8), mtype=1, mpand=(1,1), showmask=1).std.Expr('x 2 *')
scaled_a = core.std.MaskedMerge(y, scaled, line_m.resize.Point(format=vs.GRAYS), 0)
scaled_b = mdf.upscaled_sraa_mod(y, 2, 3)

scaled = fvf.rfs(scaled_a, scaled_b, mappings=f'[{opstart} {opend}]')

scaled = core.std.ShufflePlanes([scaled, den], [0, 1, 2], vs.YUV)
scaled = fvf.Depth(scaled, 16)

line_m = kgf.retinex_edgemask(scaled).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 32, 0, sample_mode=4)
db = core.std.MaskedMerge(db, scaled, line_m, 0)
db_b = dbs.f3kbilateral(db, 17, 48, 48)
db = fvf.rfs(db, db_b, mappings='[4216 4243]')

grain = kgf.adaptive_grain(db, 0.2, luma_scaling=4)

out = grain
ref = fvf.Depth(ef, 16)

cred = fvf.rfs(out, ref, mappings='[84700 87043]')

final = fvf.Depth(cred, 10)

final.set_output()
