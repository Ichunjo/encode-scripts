import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import modfunc as mdf
import mvsfunc as mvf
import debandshit as dbs
from acsuite import eztrim

core = vs.core
core.max_cache_size = 16384

src = lvf.src(r'[BDMV][191127] Fate／kaleid liner Prisma☆Illya Prisma☆Phantasm\PRISMAPHANTASM_SP\BDMV\STREAM\00005.m2ts')
asrc = r'[BDMV][191127] Fate／kaleid liner Prisma☆Illya Prisma☆Phantasm/PRISMAPHANTASM_SP/BDMV/STREAM/00005.wav'
eztrim(src, (24,-24), asrc, "fate_loli_ova_ncop.mka")

src = src[24:-24]

opstart, opend = 2789, 4946

src = fvf.Depth(src, 32)

den = mvf.BM3D(src, [1.5, 1], radius1=1)

scaled = mdf.upscaled_sraa_mod(lvf.get_y(den), 2, 3)

scaled = core.std.ShufflePlanes([scaled, den], [0, 1, 2], vs.YUV)
scaled = fvf.Depth(scaled, 16)

line_m = kgf.retinex_edgemask(scaled).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 32, 0, sample_mode=4)
db = core.std.MaskedMerge(db, scaled, line_m, 0)
db_b = dbs.f3kbilateral(db, 17, 48, 48)
db = fvf.rfs(db, db_b, mappings='[1427 1454]')

grain = kgf.adaptive_grain(db, 0.2, luma_scaling=4)

final = fvf.Depth(grain, 10)
final.set_output()
