from cooldegrain import CoolDegrain
from acsuite import eztrim
from  vsTAAmbk import TAAmbk
import lvsfunc as lvf
import fvsfunc as fvf
import modfunc as mdf
import vardefunc as vrdf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

src = lvf.src(r'[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\00003.m2ts')
eztrim(src, (396, 140436), r"[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\audio_01.w64", "audio_01_cut.mka")
eztrim(src, (396, 140436), r"[200408]「HELLO WORLD」Blu-rayスペシャル・エディション\HELLO_WORLD\BDMV\STREAM\audio_02.w64", "audio_02_cut.mka")

src = src[396:140436]
src = fvf.Depth(src, 32)

# Chroma stuff
radius = 2
y, u, v = lvf.split(src)
y_m = core.resize.Point(y, 960, 1080, src_left=-1)
y_m = core.resize.Bicubic(y_m, 960, 540)

def Demangle(c):
    return core.resize.Bicubic(c, 1920, 1080, src_left=0.25)

y_m, u, v = map(Demangle, (y_m, u, v))

y_fixup = core.std.MakeDiff(y, y_m)
yu, yv = vrdf.Regress(y_m, u, v, radius=radius)

u_fixup = vrdf.ReconstructMulti(y_fixup, yu, radius=radius)
u_r = core.std.MergeDiff(u, u_fixup)

v_fixup = vrdf.ReconstructMulti(y_fixup, yv, radius=radius)
v_r = core.std.MergeDiff(v, v_fixup)

scaled = lvf.join([y, u_r, v_r])

den = core.knlm.KNLMeansCL(scaled, a=2, h=0.6, d=3, device_type='gpu', channels='UV')
aa = TAAmbk(fvf.Depth(den, 16), aatype=0, aatypeu='Eedi3', aatypev='Eedi3', mtype=2)
warp = xvs.WarpFixChromaBlend(aa, thresh=64, blur=3)

# Luma
den = CoolDegrain(warp, tr=2, thsad=60, blksize=8, overlap=4, plane=0)
aa = lvf.nneedi3_clamp(den, ret_mask=True)

line_m = mdf.retinex_edgemask_mod(aa).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(aa, 17, 24, 24, 24, 12, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, line_m)

full = vrdf.to444(fvf.Depth(src, 16))
final = fvf.rfs(db, full, mappings='[132454 140039]')

final = fvf.Depth(final, 10)

final.set_output()