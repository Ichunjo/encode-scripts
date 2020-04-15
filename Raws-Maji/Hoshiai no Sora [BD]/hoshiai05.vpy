from acsuite import eztrim
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk
import lvsfunc as lvf
import kagefunc as kgf
import atomchtools as atf
import muvsfunc as muvf
import fvsfunc as fvf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _sraa(clip: vs.VideoNode, rep: int=13) -> vs.VideoNode:
    nnargs = dict(nsize=4, nns=3, qual=1)
    eeargs = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)

    y = lvf.get_y(clip)

    up_y = core.nnedi3cl.NNEDI3CL(y, 0, True, True, **nnargs)
    up_y = core.resize.Spline64(up_y, src_top=.5, src_left=.5)

    aa_y = core.std.Transpose(up_y)
    aa_y = core.eedi3m.EEDI3(aa_y, 0, False, **eeargs, sclip=core.nnedi3cl.NNEDI3CL(aa_y, 0, False, **nnargs))
    aa_y = core.std.Transpose(aa_y)
    aa_y = core.eedi3m.EEDI3(aa_y, 0, False, **eeargs, sclip=core.nnedi3cl.NNEDI3CL(aa_y, 0, False, **nnargs))

    scaled = core.resize.Spline64(aa_y, clip.width, clip.height)
    scaled = core.rgvs.Repair(scaled, up_y.resize.Spline64(scaled.width, scaled.height), rep)
    return core.std.ShufflePlanes([scaled, clip], [0, 1, 2], vs.YUV)

src = lvf.src('[BDMV][200226] HOSHIAI NO SORA Vol.1/BDMV/BDMV/STREAM/00006.m2ts')
asrc = '[BDMV][200226] HOSHIAI NO SORA Vol.1/BDMV/BDMV/STREAM/00006.mka'
eztrim(src, (0, -26), asrc, "[BDMV][200226] HOSHIAI NO SORA Vol.1/05_cut.mka")

src = src[:-26]
opstart, opend = 864, 3016
title = '[2854 3016]'


src = fvf.Depth(src, 16)
shift = core.fmtc.resample(src, sx=[0, -0.5], kernel='spline64')
den_uv = core.knlm.KNLMeansCL(shift, a=2, h=0.45, d=3, device_type='gpu', channels='UV')

den = CoolDegrain(den_uv, tr=1, thsad=60, blksize=8, overlap=4, plane=0)

sharp = xvs.ssharp(den, chroma=False, mask=False)
sharp = core.std.Merge(den, sharp, [.4, 0])

y = lvf.get_y(sharp)
aa_m = core.std.Sobel(y).std.Binarize(8500).std.Maximum().std.Inflate()
sq_m = kgf.squaremask(y, 260, 50, 835, 655).std.Inflate()
aa_m_f = muvf.AnimeMask(src, expr='2 *', mode=-1).rgvs.RemoveGrain(3)

aa = core.std.MaskedMerge(den, _sraa(sharp, 13), aa_m, 0)
aa_full = TAAmbk(src, aatype='Nnedi3', nsize=4, nns=4, mclip=aa_m_f)

aa = fvf.rfs(aa, aa_full, mappings=f'[{opstart} {opend}]')
aa = fvf.rfs(aa, core.std.MaskedMerge(aa_full, den, sq_m), mappings=title)


db_mask = atf.retinex_edgemask(aa, draft=True, opencl=True).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, db_mask)
grain = kgf.adaptive_grain(db, 0.2)

final = fvf.Depth(grain, 10)

final.set_output()
