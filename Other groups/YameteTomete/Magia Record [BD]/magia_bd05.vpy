"""
Magia 05
"""
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
from G41Fun import Tweak
from acsuite import eztrim
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1):
    bits = clip.format.bits_per_sample - 8
    thr = strength * (1 >> bits)

    y = kgf.get_y(clip)

    strong = TAAmbk(y, aatype='Eedi3', alpha=0.25, beta=0.5, gamma=40, nrad=2,
                    mdis=20, mtype=0, opencl=False)
    weak = TAAmbk(y, aatype='Nnedi3', nsize=3, nns=3, qual=1, mtype=0, opencl=False)
    expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

    clip_aa = core.std.Expr([strong, weak, y], expr)
    return core.std.ShufflePlanes([clip_aa, clip], [0, 1, 2], vs.YUV)

path = '[BDMV][200408][Magia Record][Vol.2]/BD_VIDEO/BDMV/STREAM/00002'
src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
blank = "blank.wav"

eztrim(src, (0, -48), asrc, "magia_bd05a.mka")
eztrim(src, (0, 117), blank, "magia_bd05b.mka")

src = src[:-48]
src = fvf.Depth(src, 16)
edstart, edend = 29374, 31530

den_a = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=4)
den_b = CoolDegrain(src, tr=3, thsad=96, blksize=8, overlap=4, plane=4)
den = lvf.rfs(den_a, den_b, [(edstart+1870, edstart+1900)])

aa_a = _nneedi3_clamp(den)
aa_b = TAAmbk(den, aatype='Eedi3')
aa = lvf.rfs(aa_a, aa_b, [(13611, 13661)])

preden = CoolDegrain(aa, tr=1, thsad=96, blksize=8, overlap=4, plane=0)
line_m = core.std.Expr([lvf.denoise.detail_mask(preden, rad=2, radc=2, brz_a=3250, brz_b=1250),
                        kgf.retinex_edgemask(preden).std.Binarize(9250).std.Median().std.Inflate()],
                       'x y max')

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 12, 0, sample_mode=4, keep_tv_range=True)
db = core.std.MaskedMerge(db, aa, line_m)
db_b = core.neo_f3kdb.Deband(db, 18, 48, 48, 48, 0, 0, sample_mode=2, keep_tv_range=True)

db_c = core.neo_f3kdb.Deband(db, 18, 64, 64, 64, 18, 0)
db_c = core.std.MaskedMerge(db_c, aa, line_m)

db = lvf.rfs(db, db_b, [(edstart+1870, edstart+1900)])
db = lvf.rfs(db, db_c, [(13611, 13661)])

grain = kgf.adaptive_grain(db, 0.25, luma_scaling=10)

crop = core.std.CropRel(grain, left=240, top=0, right=240, bottom=0)
bd = core.std.AddBorders(crop, left=240, top=0, right=240, bottom=0)
bd = lvf.rfs(grain, bd, [(0, 707), (7317, 7364), (7506, 7559), (7695, 7724)])

# Endcard
endcard = lvf.src('[BDMV][200408][Magia Record][Vol.2]/Scans/endcard5_front.png')
endcard = core.std.AssumeFPS(endcard, src)
endcard = core.std.CropRel(endcard, left=14, top=14, right=24, bottom=23)

endcard = core.resize.Spline36(endcard, 1920, 1080, vs.RGBS, dither_type='error_diffusion')
endcard = core.w2xc.Waifu2x(endcard, noise=2, scale=1, photo=True)


src_f = core.register_format(src.format.color_family, vs.FLOAT, 32,
                             src.format.subsampling_w, src.format.subsampling_h)
endcard = core.resize.Bicubic(endcard, format=src_f, matrix_s='709', dither_type='error_diffusion')

endcard = Tweak(endcard, sat=1.2, bright=-0.05, cont=1.2)


final = core.std.Splice([bd, endcard * 117], True)
final = core.resize.Bicubic(final, format=vs.YUV420P10, dither_type='error_diffusion')
final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

final.set_output(0)
