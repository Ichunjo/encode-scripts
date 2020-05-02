"""
    Magia 01
"""
from functools import partial
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
from modfunc import adptvgrnMod_mod
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
    weak = TAAmbk(y, aatype='Nnedi3', nsize=3, nns=3, qual=1, mtype=0, opencl=True)
    expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

    clip_aa = core.std.Expr([strong, weak, y], expr)
    return core.std.ShufflePlanes([clip_aa, clip], [0, 1, 2], vs.YUV)

path = '[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00001'
src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
blank = "blank.wav"
eztrim(src, (0, -48), asrc, "magia_bd01a.mka")
eztrim(src, (0, 119), blank, "magia_bd01b.mka")


src = src[:-48]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=4)

aa_a = _nneedi3_clamp(den)
aa_b = lvf.sraa(den, rep=6, sharp_downscale=False)
aa = lvf.rfs(aa_a, aa_b, [(29453, 29476), (29510, 29532), (29640, 29663),
                          (29775, 29798), (29866, 29889), (30011, 30034)])

preden = CoolDegrain(aa, tr=1, thsad=96, blksize=8, overlap=4, plane=0)
line_m = core.std.Expr([lvf.denoise.detail_mask(preden, rad=2, radc=2, brz_a=3250, brz_b=1250),
                        kgf.retinex_edgemask(preden).std.Binarize(9250).std.Median().std.Inflate()],
                       'x y max')

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 12, 0, sample_mode=4, keep_tv_range=True)
db = core.std.MaskedMerge(db, aa, line_m)

grain_a = kgf.adaptive_grain(db, 0.25, luma_scaling=10)
grain_b = adptvgrnMod_mod(db, 1, size=1.5, luma_scaling=0)
grain = lvf.rfs(grain_a, grain_b, [(5149, 5598), (8691, 10137)])

crop = core.std.Crop(grain, 240, 240, 0, 0)
borders = core.std.AddBorders(crop, 240, 240, 0, 0)
borders = lvf.rfs(grain, borders, [(5149, 5598)])


# Endcard #
endcard = lvf.src('[BDMV][200304][Magia Record][Vol.1]/Scans/endcard1_front.png')
endcard = core.std.AssumeFPS(endcard, src)
endcard = core.std.CropRel(endcard, left=10, top=17, right=17, bottom=23)

endcard = core.resize.Bicubic(endcard, src.width, src.height, vs.RGBS, dither_type='error_diffusion')
endcard = kgf.iterate(endcard, partial(core.w2xc.Waifu2x, noise=3, scale=1, photo=True), 4)


src_f = core.register_format(src.format.color_family, vs.FLOAT, 32,
                             src.format.subsampling_w, src.format.subsampling_h)
endcard = core.resize.Bicubic(endcard, format=src_f, matrix_s='709', dither_type='error_diffusion')

endcard = Tweak(endcard, sat=1.2, bright=-0.05, cont=1.2)



final = core.std.Splice([borders, endcard * 120], True)
final = core.resize.Bicubic(final, format=vs.YUV420P10, dither_type='error_diffusion')
final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

final.set_output(0)
