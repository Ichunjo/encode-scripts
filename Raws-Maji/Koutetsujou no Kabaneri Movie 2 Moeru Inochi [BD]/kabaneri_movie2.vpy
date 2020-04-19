"""
Kabaneri Movie 2
"""
from typing import Callable
from functools import partial

from acsuite import eztrim
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
import lvsfunc as lvf
import fvsfunc as fvf
import atomchtools as atf
import vardefunc as vrf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _perform_upscaling(clip: vs.VideoNode, width: int, height: int,
                       downscaler: Callable[[vs.VideoNode], vs.VideoNode]) -> vs.VideoNode:
    def _nnedi3(clip):
        args = dict(nsize=0, nns=4, qual=2, pscrn=2)
        if hasattr(core, 'znedi3'):
            clip = clip.std.Transpose().znedi3.nnedi3(0, True, **args) \
                .std.Transpose().znedi3.nnedi3(0, True, **args)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
                .std.Transpose().nnedi3.nnedi3(0, True, **args)
        return clip

    def _nnedi3cl(clip):
        args = dict(nsize=0, nns=4, qual=2, pscrn=2)
        return clip.nnedi3cl.NNEDI3CL(0, True, True, **args)

    double = core.std.Interleave([_nnedi3(clip[::2]), _nnedi3cl(clip[1::2])])
    return downscaler(double, width, height, src_top=.5, src_left=.5)

def _transfer_prop(n, f):
    fout = f[0].copy()
    fout.props['PlaneStatsDiff'] = f[1].props['PlaneStatsDiff']
    fout.props['_b'] = f[1].props['_b']
    fout.props['_c'] = f[1].props['_c']
    return fout

def _select_best(n, f, clips):
    clip_data = []
    for p in f:
        clip_data.append(p.props['PlaneStatsDiff'])
    return clips[clip_data.index(min(clip_data))]

src = lvf.src('KABANERI_MOVIE/KABANERI_MOVIE_DISC2/BDMV/STREAM/00002.m2ts')
asrc1 = 'KABANERI_MOVIE/KABANERI_MOVIE_DISC2/BDMV/STREAM/audio1.w64'
asrc2 = 'KABANERI_MOVIE/KABANERI_MOVIE_DISC2/BDMV/STREAM/audio2.w64'
eztrim(src, (360, -24), asrc1, "KABANERI_MOVIE/movie2_audio1_cut.mka")
eztrim(src, (360, -24), asrc2, "KABANERI_MOVIE/movie2_audio2_cut.mka")


src = fvf.Depth(src[360:-24], 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)
den_b = CoolDegrain(src, tr=1, thsad=84, blksize=8, overlap=4, plane=4)
den = fvf.rfs(den, den_b, mappings='[103775 103969]')



params = [[1/5, 2/5], [1/5, 1/2], [0.3782, 0.3109], [1/3, 1/3]]
h = 916
w = lvf.get_w(h)
y = lvf.get_y(fvf.Depth(den, 32))

clips_ = []
for param in params:
    desc = core.descale.Debicubic(y, w, h, param[0], param[1])
    upsc = core.resize.Bicubic(desc, y.width, y.height, filter_param_a=param[0],
                               filter_param_b=param[1])
    upsc = core.std.PlaneStats(y, upsc) \
   	    .std.SetFrameProp('_b', floatval=param[0]) \
       	    .std.SetFrameProp('_c', floatval=param[1])
    desc = core.std.ModifyFrame(desc, [desc, upsc], _transfer_prop)
    clips_.append(desc)


descaled = core.std.FrameEval(clips_[0], partial(_select_best, clips=clips_), prop_src=clips_)
upscaled = _perform_upscaling(fvf.Depth(descaled, 16), y.width, y.height,
                              partial(core.resize.Spline64))
scaled = core.std.ShufflePlanes([upscaled, den], [0, 1, 2], vs.YUV)
aa = TAAmbk(den, aatype='Nnedi3', mtype=1, nsize=4)
scaled = fvf.rfs(scaled, aa, mappings='[140719 144422]')




line_m = atf.retinex_edgemask(scaled, draft=True, opencl=False) \
    .std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(scaled, 16, 36, 36, 36, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(db, 20, 90, 90, 90, 32, 0, sample_mode=2)
db = fvf.rfs(db, db_b, mappings='[103877 103969] [107252 107328]')

db = core.std.MaskedMerge(db, scaled, line_m)




cred_m = vrf.drm(src, h, 'bicubic', params[0][0], params[0][1], 80)
cred = fvf.rfs(db, src, mappings='[0 289]')

final = fvf.Depth(cred, 10)

final.set_output()
