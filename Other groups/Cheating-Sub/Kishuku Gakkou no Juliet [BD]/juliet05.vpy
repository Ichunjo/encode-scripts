from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import atomchtools as atf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _doubling(clip: vs.VideoNode) -> vs.VideoNode:
    args = dict(nsize=0, nns=4, qual=2, pscrn=2)

    def _nnedi3(clip):
        return clip.std.Transpose().znedi3.nnedi3(0, True, **args) \
            .std.Transpose().znedi3.nnedi3(0, True, **args)

    def _nnedi3cl(clip):
        return clip.nnedi3cl.NNEDI3CL(0, True, True, **args)

    double = core.std.Interleave([_nnedi3(clip[::2]), _nnedi3cl(clip[1::2])])
    return core.resize.Spline36(double, src_top=.5, src_left=.5)

def _sraa(clip: vs.VideoNode) -> vs.VideoNode:
    nnargs = dict(nsize=4, nns=3, qual=1, pscrn=2)
    eeargs = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)

    def _znedi3(clip):
        return clip.znedi3.nnedi3(0, False, **nnargs)

    def _eedi3(clip, zclip):
        return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=zclip)

    aa = _eedi3(clip, _znedi3(clip)).std.Transpose()
    aa = _eedi3(aa, _znedi3(aa)).std.Transpose()
    return aa

src = lvf.src(r'[BDMV][アニメ][190125][寄宿学校のジュリエット][Vol.02]/GNXA_217201/BDMV/STREAM/00005.m2ts')
src = src[:34694]

b, c = 0, 1

src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)

y = lvf.get_y(den)
y_ = fvf.Depth(y, 32)
desc = core.descale.Debicubic(y_, y.width/1.5, y.height/1.5, b, c)
upsc = core.resize.Bicubic(desc, y.width, y.height, filter_param_a=b, filter_param_b=c)

cred_m = core.std.Expr([y_, upsc], 'x y - abs').std.Binarize(0.05)
cred_m = kgf.iterate(cred_m, core.std.Maximum, 4)
cred_m = kgf.iterate(cred_m, core.std.Inflate, 2)
cred_m = fvf.Depth(cred_m, 16)

doubled = _doubling(fvf.Depth(desc, 16))
sraa = _sraa(doubled)
rep = core.rgvs.Repair(sraa, doubled, 13)

line_m = core.std.Sobel(sraa).std.Expr('x 6 *').std.Maximum().std.Minimum().std.Binarize(12500)
sangnom = sraa.sangnom.SangNom(aa=128).std.Transpose().sangnom.SangNom(aa=128).std.Transpose()
sangnom = sangnom.sangnom.SangNom(aa=128).std.Transpose().sangnom.SangNom(aa=128).std.Transpose()
sangnom = core.std.MaskedMerge(sraa, sangnom, line_m)
rep = fvf.rfs(rep, sangnom, mappings='[19010 19099]')


scaled = core.fmtc.resample(rep, y.width, y.height, kernel='gauss',
                            invks=True, invkstaps=1, taps=1, a1=32)
sharp = xvs.ssharp(scaled, mask=True)
sharp = core.std.Merge(scaled, sharp, 0.4)


cred = core.std.MaskedMerge(sharp, y, cred_m, 0)
merged = core.std.ShufflePlanes([cred, den], [0, 1, 2], vs.YUV)

aa_cred = TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=1, mpand=(1, 1))
aa_cred = core.std.MaskedMerge(merged, aa_cred, cred_m)

db_m = atf.retinex_edgemask(aa_cred, draft=True, opencl=True) \
   .std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(aa_cred, 17, 36, 36, 36, 24, 0, sample_mode=4, keep_tv_range=True)
db = core.std.MaskedMerge(db, aa_cred, db_m)

final = fvf.Depth(db, 10)

final.set_output()
