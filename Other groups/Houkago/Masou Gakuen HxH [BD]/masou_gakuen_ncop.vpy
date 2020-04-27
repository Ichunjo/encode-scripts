from cooldegrain import CoolDegrain
from acsuite import eztrim
import atomchtools as atf
import fvsfunc as fvf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 8192

def _doubling(clip: vs.VideoNode) -> vs.VideoNode:
    args = dict(nsize=0, nns=4, qual=2, pscrn=2)

    def _nnedi3(clip):
        return clip.std.Transpose().znedi3.nnedi3(0, True, **args) \
            .std.Transpose().znedi3.nnedi3(0, True, **args)

    def _nnedi3cl(clip):
        return clip.nnedi3cl.NNEDI3CL(0, True, True, **args)

    return core.resize.Spline36(
        core.std.Interleave([_nnedi3(clip[::2]), _nnedi3cl(clip[1::2])]),
        src_top=.5, src_left=.5)


path = "魔装学園H×H/MASOUGAKUEN_HH_1/BDMV/STREAM/00010."
src = lvf.src(path + "m2ts")
asrc = path + "wav"
eztrim(src, (24, -24), asrc, "masou_gakuen_ncop.mka")

src = src[24:-24]

fe = lvf.edgefixer(src, [1, 1, 1])

den = CoolDegrain(fe, tr=1, thsad=60, bits=16, blksize=8, overlap=4, plane=4)


y = lvf.get_y(fvf.Depth(den, 32))


desc_y = core.descale.Debilinear(y, 1280, 720)
upsc_y = core.resize.Bilinear(desc_y, 1920, 1080)

cred_m = core.std.Expr([y, upsc_y], 'x y - abs').std.Binarize(0.03)
cred_m = lvf.iterate(cred_m, core.std.Maximum, 4)
cred_m = lvf.iterate(cred_m, core.std.Inflate, 2)

cred_m, desc_y = [fvf.Depth(c, 16) for c in [cred_m, desc_y]]


doubled_y = _doubling(desc_y)
scaled_y = core.resize.Bicubic(doubled_y, 1920, 1080)
aa_y = lvf.upscaled_sraa(lvf.get_y(den), 1.85, 3)
scaled_y = lvf.rfs(scaled_y, core.std.MaskedMerge(scaled_y, aa_y, cred_m), [(387, 779), (1054, 1186)])

merged = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

merged = fvf.Depth(merged, 32)


den_clip = CoolDegrain(merged, tr=2, thsad=120, bits=16, blksize=8, overlap=4, plane=4)
den_clip = fvf.Depth(den_clip, 32)

db_ma = atf.retinex_edgemask(den_clip, draft=True, opencl=True).rgvs.RemoveGrain(4).std.Expr('x 2 *')
db_ma = fvf.Depth(db_ma, 32)

db_mb = core.std.Prewitt(den_clip).std.Binarize(0.0325) \
    .rgsf.RemoveGrain(4).std.Maximum().std.Minimum()
db_mb = lvf.get_y(db_mb)
db_m = core.std.Expr([db_ma, db_mb], 'x y +')

db_a = core.placebo.Deband(merged, radius=17, threshold=5.5, iterations=1, grain=0, planes=1|2|4)
db_b = core.placebo.Deband(merged, radius=17, threshold=8, iterations=2, grain=0, planes=1|2|4)
db = lvf.rfs(db_a, db_b, [(130, 143), (994, 1053)])
db = core.std.MaskedMerge(db_a, merged, db_m)

grain = core.placebo.Deband(db_a, radius=0, threshold=0, iterations=0, grain=6.75, planes=1)

final = core.fmtc.bitdepth(grain, bits=10, dmode=3)
final.set_output()
