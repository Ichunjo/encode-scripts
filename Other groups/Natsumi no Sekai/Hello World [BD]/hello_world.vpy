from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk
from acsuite import eztrim
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrf
import atomchtools as atf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1, mask: vs.VideoNode = None):
    bits = clip.format.bits_per_sample - 8
    thr = strength * (1 >> bits)

    y = lvf.get_y(clip)

    strong = TAAmbk(y, aatype='Eedi3', alpha=0.25, beta=0.5, gamma=200, nrad=2, 
                    mdis=20, mtype=0, opencl=False)
    weak = TAAmbk(y, aatype='Nnedi3', nsize=3, nns=3, qual=1, mtype=0, opencl=True)
    expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

    aa = core.std.Expr([strong, weak, y], expr)
    merged = core.std.MaskedMerge(y, aa, mask)
    return core.std.ShufflePlanes([merged, clip], [0, 1, 2], vs.YUV)

src = lvf.src(r'[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\00003.m2ts')
eztrim(src, (396,140436), r"[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\audio_01.w64", "audio_01_cut.mka")
eztrim(src, (396,140436), r"[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\audio_02.w64", "audio_02_cut.mka")
eztrim(src, (396,140436), r"[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\audio_03.w64", "audio_03_cut.mka")

src = src[396:140436]
src = fvf.Depth(src, 16)

den = core.knlm.KNLMeansCL(src, a=2, h=0.6, d=3, device_type='gpu', channels='UV')

# Chroma stuff
radius = 3
y, u, v = [fvf.Depth(plane, 32) for plane in lvf.split(den)]


y_m = core.resize.Point(y, 960, 1080, src_left=-1)
y_m = core.resize.Bicubic(y_m, 960, 540)

def _demangle(c):
    args = dict(nsize=1, nns=4, qual=2)
    c = c.std.Transpose().nnedi3.nnedi3(0, True, **args) \
              .std.Transpose().nnedi3.nnedi3(0, True, **args)
    return core.resize.Bicubic(c, 1920, 1080, src_top=.5, src_left=.5)

y_m, u, v = map(_demangle, (y_m, u, v))

y_fixup = core.std.MakeDiff(y, y_m)
yu, yv = vrf.Regress(y_m, u, v, radius=radius)

u_fixup = vrf.ReconstructMulti(y_fixup, yu, radius=radius)
u_r = core.std.MergeDiff(u, u_fixup)

v_fixup = vrf.ReconstructMulti(y_fixup, yv, radius=radius)
v_r = core.std.MergeDiff(v, v_fixup)

scaled = lvf.join([fvf.Depth(plane, 16) for plane in [y, u_r, v_r]])

# Luma
den = CoolDegrain(scaled, tr=1, thsad=60, blksize=8, overlap=4, plane=0)

line_m = atf.retinex_edgemask(den, draft=True, opencl=True)
aa = _nneedi3_clamp(den, mask=line_m)

db = core.neo_f3kdb.Deband(aa, 17, 24, 24, 24, 12, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

full = vrf.to444(src)
final = fvf.rfs(db, full, mappings='[132454 140039]')

final = fvf.Depth(final, 10)

final.set_output()
