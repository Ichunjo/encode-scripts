from functools import partial
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import atomchtools as atf
import xvs
import mvsfunc as mvf
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

def _select_best(n, f, clips):
    clip_data = []
    for p in f:
        clip_data.append(p.props.PlaneStatsAverage)
    return clips[clip_data.index(min(clip_data))]

src = lvf.src(r'[BDMV][アニメ][190222][寄宿学校のジュリエット][Vol.03]/GNXA_217301/BDMV/STREAM/00006.m2ts')

src = src[:34335]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)

y = lvf.get_y(den)
y_ = fvf.Depth(y, 32)

# Opening and endings are bicubic sharp
# The episode is catmull-rom but >scenefiltering and there is a lot of signs
# so automating that is better.
params = [[0, 1/2], [0, 1]]
desc_clips_, upsc_clips_, smask_clips_ = [], [], []

for param in params:
    desc = core.descale.Debicubic(y_, y.width/1.5, y.height/1.5, param[0], param[1])
    upsc = core.resize.Bicubic(desc, y.width, y.height, filter_param_a=param[0],
                               filter_param_b=param[1])
    # Can be commented.
    # It's just here to check the kernel used
    upsc, desc = [c.std.SetFrameProp('Param_b', floatval=param[0]) \
        .std.SetFrameProp('Param_c', floatval=param[1])
                  for c in [upsc, desc]]

    # Same approach than getnative
    # Don't ask me what this expression does
    # I have no idea
    smask = core.std.Expr([y_, upsc], "x y - abs dup 0.015 > swap 0 ?")
    smask = core.std.CropRel(smask, 5, 5, 5, 5).std.PlaneStats()

    desc_clips_.append(desc)
    upsc_clips_.append(upsc)
    smask_clips_.append(smask)

# Selecting the best kernel according to the PlaneStatsAverage of the smask of the current frame
# We use the upscaled clip after to generate the credit mask
descaled, upscaled = [core.std.FrameEval(clips_[0], partial(_select_best, clips=clips_),
                                         prop_src=smask_clips_)
                      for clips_ in [desc_clips_, upsc_clips_]]

# Standart credit mask
cred_m = core.std.Expr([y_, upscaled], 'x y - abs').std.Binarize(0.05)
cred_m = kgf.iterate(cred_m, core.std.Maximum, 4)
cred_m = kgf.iterate(cred_m, core.std.Inflate, 2)

descaled, cred_m = [fvf.Depth(c, 16) for c in [descaled, cred_m]]



doubled = _doubling(descaled)
sraa = _sraa(doubled)
rep = core.rgvs.Repair(sraa, doubled, 13)


# Sharp downscale to compensate the sraa and because it looks better
scaled = core.fmtc.resample(rep, y.width, y.height, kernel='gauss', invks=True, invkstaps=1, taps=1, a1=32)
sharp = xvs.ssharp(scaled, mask=True)
sharp = core.std.Merge(scaled, sharp, 0.4)


cred = core.std.MaskedMerge(sharp, y, cred_m, 0)
merged = core.std.ShufflePlanes([cred, den], [0, 1, 2], vs.YUV)


# Very mild AA for 1080p stuff
aa_cred = TAAmbk(den, aatype='Nnedi3', nsize=4, nns=4, mtype=1, mpand=(1, 1))
aa_cred = core.std.MaskedMerge(merged, aa_cred, cred_m)

# Diff mask for large gradients when even the large amount of grain
# present in the source isn't sufficient to cover the banding
hard_den = mvf.BM3D(aa_cred, [5, 2.5])
diff = core.std.MakeDiff(aa_cred, hard_den)

db_m = atf.retinex_edgemask(aa_cred, draft=True, opencl=True) \
   .std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(aa_cred, 17, 36, 36, 36, 24, 0, sample_mode=4, keep_tv_range=True)

db_b = core.neo_f3kdb.Deband(hard_den, 17, 42, 42, 42, 24, 0, sample_mode=4, keep_tv_range=True)
db_b = core.std.MergeDiff(db_b, diff)
db = fvf.rfs(db, db_b, '[31217 31345]')

db = core.std.MaskedMerge(db, aa_cred, db_m)


final = fvf.Depth(db, 10)

final.set_output()
