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

path = r"[BDMV] Ao-chan Can't Study!\Ao-chan Cant Study BD-1\BDMV\STREAM\00053"
src = lvf.src(path + '.m2ts')
eztrim(src, (24, 17616), path + '.wav', "aochan_04.mka")

src = src[24:17616]

h = 810
w = lvf.scale.get_w(h)
taps = 5
opstart, opend = 526, 1963
edstart, edend = 15058, 17357
full_stuff = [(1976, 2095), (17490, src.num_frames - 1)]

den = CoolDegrain(fvf.Depth(src, 16), tr=1, thsad=60, blksize=8, overlap=4, plane=4)


y, u, v = vrf.split(fvf.Depth(den, 32))


desc = core.descale.Delanczos(y, w, h, taps)
upsc = core.resize.Lanczos(desc, y.width, y.height, filter_param_a=taps)

aa = _nnedi3(desc, w, h, (0, 0), (.5, .5))
from vsTAAmbk import TAAmbk
aa_b = lvf.sraa(desc, 1.35, rep=6)
aa_b = lvf.util.quick_resample(aa_b, TAAmbk, aatype='Eedi3')
aa = lvf.rfs(aa, aa_b, [(10896, 11007), (11189, 11407)])
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
db_b = core.placebo.Deband(db, radius=20, threshold=8, iterations=1, grain=0, planes=2|4)
db = lvf.rfs(db, db_b, [(13880, 13920), (14222, 14437)])

db = core.std.MaskedMerge(db, out, line_m)

grain = core.std.ShufflePlanes(
    [kgf.adaptive_grain(vrf.get_y(db), 0.2, luma_scaling=8), db],
    [0, 1, 2], vs.YUV
)
from modfunc import adptvgrnMod_mod
grain_b = adptvgrnMod_mod(db, 1, size=2, luma_scaling=0, static=False)
grain = lvf.rfs(grain, grain_b, [(0, 525)])


final = core.resize.Point(grain, format=vs.YUV444P10, dither_type='error_diffusion')

final.set_output(0)
