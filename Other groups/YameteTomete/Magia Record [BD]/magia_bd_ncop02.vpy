"""
Magia NCOP02
"""
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
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

path = '[BDMV][200408][Magia Record][Vol.2]/BD_VIDEO/BDMV/STREAM/00004'

src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
eztrim(src, (24, -24), asrc, "magia_bd_ncop02.wav")

src = src[24:-24]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=4)


aa = _nneedi3_clamp(den)

preden = CoolDegrain(aa, tr=1, thsad=96, blksize=8, overlap=4, plane=0)
line_m = core.std.Expr([lvf.denoise.detail_mask(preden, rad=2, radc=2, brz_a=3250, brz_b=1250),
                        kgf.retinex_edgemask(preden).std.Binarize(9250).std.Median().std.Inflate()],
                       'x y max')

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 12, 0, sample_mode=4, keep_tv_range=True)
db = core.std.MaskedMerge(db, aa, line_m)

grain = kgf.adaptive_grain(db, 0.25, luma_scaling=10)

final = core.resize.Bicubic(grain, format=vs.YUV420P10, dither_type='error_diffusion')
final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

final.set_output(0)
