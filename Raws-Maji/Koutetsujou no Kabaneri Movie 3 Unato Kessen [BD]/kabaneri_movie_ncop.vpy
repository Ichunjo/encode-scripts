from typing import List
from functools import partial
from acsuite import eztrim
from cooldegrain import CoolDegrain
from nnedi3_rpow2CL import nnedi3_rpow2CL
import lvsfunc as lvf
import fvsfunc as fvf
import modfunc as mdf
import vardefunc as vrf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

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

src = lvf.src('[191211][BDMV] 甲鉄城のカバネリ 海門決戦/KABANERI_UNATO_BONUS/BDMV/STREAM/00004.m2ts')
asrc = '[191211][BDMV] 甲鉄城のカバネリ 海門決戦/KABANERI_UNATO_BONUS/BDMV/STREAM/ncop.w64'
eztrim(src, (24, -24), asrc, "[191211][BDMV] 甲鉄城のカバネリ 海門決戦/audio_ncop_cut.mka")


src = fvf.Depth(src[24:-24], 16)

den = CoolDegrain(src, tr=1, thsad=60, blksize=8, overlap=4, plane=4)
den = fvf.Depth(den, 32)

params: List[List[int]] = [[1/5, 2/5], [1/5, 1/2], [0.3782, 0.3109], [1/3, 1/3]]
y = lvf.get_y(den)

clips_ = []
for param in params:
    desc = core.descale.Debicubic(y, lvf.get_w(916), 916, param[0], param[1])
    upsc = core.resize.Bicubic(desc, y.width, y.height,
                               filter_param_a=param[0], filter_param_b=param[1])
    upsc = core.std.PlaneStats(y, upsc)
    upsc = core.std.SetFrameProp(upsc, '_b', floatval=param[0]) \
        .std.SetFrameProp('_c', floatval=param[1])
    desc = core.std.ModifyFrame(desc, [desc, upsc], _transfer_prop)
    clips_.append(desc)


descaled = core.std.FrameEval(clips_[0], partial(_select_best, clips=clips_), prop_src=clips_)
upscaled = nnedi3_rpow2CL(descaled, nns=4, qual=2).resize.Spline64(y.width, y.height)
scaled = core.std.ShufflePlanes([upscaled, den], [0, 1, 2], vs.YUV)
scaled = fvf.Depth(scaled, 16)

line_m = mdf.retinex_edgemask_mod(scaled).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db_a = core.neo_f3kdb.Deband(scaled, 16, 36, 36, 36, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(scaled, 16, 54, 54, 54, 32, 0, sample_mode=4)
db = fvf.rfs(db_a, db_b, mappings='[158 186]')


db = core.std.MaskedMerge(db, scaled, line_m)

cred_m = vrf.drm(src, 916, 'bicubic', params[0][0], params[0][1], 80)
cred = core.std.MaskedMerge(db, fvf.Depth(den, 16), cred_m, 0)

final = fvf.Depth(cred, 10)

final.set_output()
