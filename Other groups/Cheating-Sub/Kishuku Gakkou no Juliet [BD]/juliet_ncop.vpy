from cooldegrain import CoolDegrain
from acsuite import eztrim
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

src = lvf.src(r'[BDMV][アニメ][181222][寄宿学校のジュリエット][Vol.01]/GNXA_217101/BDMV/STREAM/00014.m2ts')
#asrc = r"BRMM_10243BD/STREAM/00001.wav"
#ac.eztrim(src, [(0,34047)], asrc, "bangdreamS03E10.mka")


src = src[:-24]

b, c = 0, 1

src = fvf.Depth(src, 16)
den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)

y = lvf.get_y(den)
y_ = fvf.Depth(y, 32)
desc = core.descale.Debicubic(y_, y.width/1.5, y.height/1.5, b, c)
upsc = core.resize.Bicubic(desc, y.width, y.height, filter_param_a=b, filter_param_b=c)

doubled = _doubling(fvf.Depth(desc, 16))
sraa = _sraa(doubled)
rep = core.rgvs.Repair(sraa, doubled, 13)


scaled = core.fmtc.resample(rep, y.width, y.height, kernel='gauss',
                            invks=True, invkstaps=1, taps=1, a1=32)
sharp = xvs.ssharp(scaled, mask=True)
sharp = core.std.Merge(scaled, sharp, 0.4)


merged = core.std.ShufflePlanes([sharp, den], [0, 1, 2], vs.YUV)

db_m = atf.retinex_edgemask(merged, draft=True, opencl=True) \
   .std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(merged, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, merged, db_m)

final = fvf.Depth(db, 10)

final.set_output()
