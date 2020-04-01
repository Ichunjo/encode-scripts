import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import muvsfunc as muvf
import xvs
import acsuite
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = lvf.src(r'VIOLET_EVERGARDEN_SIDESTORY\BDMV\STREAM\00000.m2ts')
asrc = r'VIOLET_EVERGARDEN_SIDESTORY\BDMV\STREAM\audio1.w64'
ac.eztrim(src,[(681,-24)], asrc,"audio1.mka")
asrc = r'VIOLET_EVERGARDEN_SIDESTORY\BDMV\STREAM\audio2.w64'
ac.eztrim(src,[(681,-24)], asrc,"audio2.mka")
asrc = r'VIOLET_EVERGARDEN_SIDESTORY\BDMV\STREAM\commentary.w64'
ac.eztrim(src,[(681,-24)], asrc,"commentary.mka")

src = src[681:-24]

crop = core.std.Crop(src, 0, 0, 138, 138)

shift = core.fmtc.resample(crop, sx=[0, 0.25], kernel='spline64')
den_uv = core.knlm.KNLMeansCL(shift, a=2, h=0.45, d=3, device_type='gpu', channels='UV')

warp = xvs.WarpFixChromaBlend(den_uv, thresh=128, blur=2)

den_y = CoolDegrain(warp, tr=1, thsad=48, blksize=8, overlap=4, plane=0)

db = core.neo_f3kdb.Deband(den_y, 16, 24, 24, 24, 24, 0, sample_mode=4)
grain = kgf.adaptive_grain(db, 0.25)


final = fvf.rfs(grain, fvf.Depth(crop, 16), mappings='[122454 129710]')
final = core.fmtc.bitdepth(final, bits=10, dmode=3)

credits = core.std.AssumeFPS(final, fpsnum=12000, fpsden=1001)
smooth = core.std.Interleave([
			credits, 
			core.resize.Spline36(credits, src_top=3/5),
			core.resize.Spline36(credits, src_top=(3/5)*2),
			core.resize.Spline36(credits, src_top=(3/5)*3),
			core.resize.Spline36(credits, src_top=(3/5)*4)
			]).std.AssumeFPS(fpsnum=120000, fpsden=1001)
				
				
vfr = muvf.VFRSplice([final[:122664], smooth[613320:647840], final[129568:final.num_frames-1]], 'violet_movie1_timecode.txt', precision=6)

vfr.set_output()