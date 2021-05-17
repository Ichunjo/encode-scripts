from acsuite import eztrim
import mvsfunc as mvf
import havsfunc as hvf
import vardefunc as vrf
import fvsfunc as fvf
import kagefunc as kgf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 1024*32

def from_16_to_32(var):
    return (var - (16 << 8)) / ((235 << 8) - (16 << 8))

def _eedi3_instance(clip, dh, eeargs, nnargs) -> vs.VideoNode:
    return core.eedi3m.EEDI3(clip, 0, dh, **eeargs, sclip=_nnedi3_sclip(clip, dh, nnargs))

def _nnedi3_sclip(clip, dh, nnargs) -> vs.VideoNode:
    return core.nnedi3.nnedi3(clip, 0, dh, **nnargs)

def _rescale(clip, width, height, eeargs, nnargs) -> vs.VideoNode:
    clip = _eedi3_instance(clip, True, eeargs, nnargs).std.Transpose()
    clip = _eedi3_instance(clip, True, eeargs, nnargs).std.Transpose()
    mask = clip.std.Prewitt().std.Binarize(from_16_to_32(8500)).std.Maximum()
    clip = core.std.MaskedMerge(clip, _sraa(clip), mask)
    return core.resize.Spline36(clip, width, height, src_left=.5, src_top=.5)

def _sraa(clip: vs.VideoNode) -> vs.VideoNode:
    eeargs = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    nnargs = dict(nsize=0, nns=4, qual=2)
    clip = _eedi3_instance(clip, False, eeargs, nnargs).std.Transpose()
    clip = _eedi3_instance(clip, False, eeargs, nnargs).std.Transpose()
    return clip


src = lvf.src(r"[BDMV] We Never Learn BOKUBEN OAD 2\BD_VIDEO\BDMV\STREAM\00001.m2ts")
eztrim(src, (24, -24), r"[BDMV] We Never Learn BOKUBEN OAD 2\BD_VIDEO\BDMV\STREAM\00001.mka", "bokuben_ova02.mka", mkvextract_path="mkvextract")
src = src[24:-24]

fe = lvf.misc.edgefixer(src, [1, 1, 1])

den = mvf.BM3D(fe, [1, 0.75], 1, depth=32)
planes = vrf.split(den)

planes[0] = core.descale.Debilinear(planes[0], 1536, 864)


eedi3_args = dict(alpha=.3, beta=.4, gamma=1000, mdis=20, vcheck=3)
nnedi3_args = dict(nns=4, nsize=2, qual=2, pscrn=1)
planes[0] = _rescale(planes[0], 1920, 1080, eedi3_args, nnedi3_args)

rescaled = fvf.Depth(vrf.join(planes), 16)

dering = hvf.EdgeCleaner(rescaled, 5, rmode=1, smode=1, hot=True)

db_m = lvf.denoise.detail_mask(dering, brz_a=2500, brz_b=1000)
db = core.neo_f3kdb.Deband(dering, 17, 36, 36, 36, 24, 0, sample_mode=4, keep_tv_range=True)
db = core.std.MaskedMerge(db, dering, db_m)

grain = kgf.adaptive_grain(db, 0.3, luma_scaling=8)


cred_m = vrf.drm(src, 864, 'bilinear', mthr=70)
cred = lvf.rfs(grain, core.std.MaskedMerge(grain, fvf.Depth(src, 16), fvf.Depth(cred_m, 16)),
               [(5561, 5634), (5771, 5912), (13531, 13637), (17025, 17083), (30707, 32860)])

final = core.resize.Point(grain, format=vs.YUV420P10, dither_type="error_diffusion")
final = core.std.Limiter(final, 16 << 2, [235 << 2, 240 << 2])

final.set_output(0)
