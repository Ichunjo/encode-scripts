from typing import Tuple
from acsuite import eztrim
from cooldegrain import CoolDegrain
import kagefunc as kgf
import vardefunc as vrf
import fvsfunc as fvf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _nnedi3(clip: vs.VideoNode, width: int, height: int,
            field: Tuple[float, float], shift: Tuple[float, float], **args):
    args = args or dict(nsize=4, nns=4, qual=2)
    return core.resize.Spline36(
        clip.std.Transpose().nnedi3.nnedi3(field[0], True, **args) \
            .std.Transpose().nnedi3.nnedi3(field[1], True, **args),
        width, height, src_top=shift[0], src_left=shift[1]
    )

path = r"[BDMV] Ao-chan Can't Study!\Ao-chan Cant Study BD-1\BDMV\STREAM\00006"
src = lvf.src(path + '.m2ts')
eztrim(src, (24, 17616), path + '.wav', "aochan_01.mka")

src = src[24:17616]

h = 810
w = lvf.scale.get_w(h)
taps = 5
opstart, opend = 744, 2181
edstart, edend = 15920, 17357
full_stuff = [(2194, 2337), (17506, src.num_frames - 1)]

src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)
den = lvf.rfs(den, src, [(3881, 3988), (4515, 4562)])

y, u, v = vrf.split(fvf.Depth(den, 32))


desc = core.descale.Delanczos(y, w, h, taps)
upsc = core.resize.Lanczos(desc, y.width, y.height, filter_param_a=taps)


aa = _nnedi3(desc, w, h, (0, 0), (.5, .5))
u, v = map(lambda x: _nnedi3(x, w, h, (1, 0), (.5, 0)), (u, v))


# Perform credit mask for OP / ED / Random shits
cred_m = core.std.Expr([y, upsc], 'x y - abs').std.Binarize(0.05)
cred_m = vrf.iterate(cred_m, core.std.Maximum, 4)
cred_m = vrf.iterate(cred_m, core.std.Inflate, 2)

cred_m, y = [core.resize.Spline36(c, w, h) for c in [cred_m, y]]


cred = lvf.rfs(aa, core.std.MaskedMerge(aa, y, cred_m, 0),
               [(opstart, opend), (edstart, edend)] + full_stuff)


full = vrf.join([cred, u, v])
out = fvf.Depth(full, 16)


line_m = lvf.denoise.detail_mask(out.std.Median(), brz_a=1850, brz_b=875)
db = core.neo_f3kdb.Deband(out, 15, 36, 36, 36, 24, 0, sample_mode=4, dynamic_grain=True)
db = core.std.MaskedMerge(db, out, line_m)


from mvsfunc import BM3D
preden = BM3D(out, 7.5, 1)
diff = core.std.MakeDiff(out, preden)
line_m = kgf.retinex_edgemask(preden).std.Binarize(9800)
line_m = vrf.iterate(line_m, core.std.Maximum, 2)
line_m = vrf.iterate(line_m, core.std.Inflate, 2)
db_b = core.placebo.Deband(preden, radius=17, threshold=10, iterations=3, grain=0, planes=1|2|4)
db_b = core.std.MaskedMerge(db_b, out, line_m)
db_b = core.std.MergeDiff(db_b, diff)

db = lvf.rfs(db, db_b, [(3881, 3988), (8345, 8448)])


grain = core.std.ShufflePlanes(
    [kgf.adaptive_grain(vrf.get_y(db), 0.2, luma_scaling=8), db],
    [0, 1, 2], vs.YUV
)

final = core.resize.Point(grain, format=vs.YUV444P10, dither_type='error_diffusion')

final.set_output(0)
