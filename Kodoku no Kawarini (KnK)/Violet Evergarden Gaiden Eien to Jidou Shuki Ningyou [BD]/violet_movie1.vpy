import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import muvsfunc as muvf
import xvs
import acsuite
from cooldegrain import CoolDegrain
from vsutils import vsutils

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()
vstool = vsutils()

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

# # # # # # # # #
credits = core.std.AssumeFPS(final, fpsnum=12000, fpsden=1001)
smooth = core.std.Interleave([
    credits,
    core.resize.Spline36(credits, src_top=3 / 5),
    core.resize.Spline36(credits, src_top=(3 / 5) * 2),
    core.resize.Spline36(credits, src_top=(3 / 5) * 3),
    core.resize.Spline36(credits, src_top=(3 / 5) * 4)
]).std.AssumeFPS(
    fpsnum=120000, fpsden=1001)

smooth = core.resize.Point(smooth, src_left=-360)

# # # # # # # # #
start_frame = 616224
last_frame = 647840
length = (last_frame - start_frame) / 26

ref = core.std.BlankClip(smooth, lvf.get_w(342), 342)
clip = ref[:1]

for i in range(1, 27):
	image = core.imwri.Read(f'Screenshots/Screenshot_({i}).png', float_output=True)
	image = core.std.AssumeFPS(image, fpsnum=ref.fps.numerator, fpsden=ref.fps.denominator)
	image = core.resize.Spline36(image, ref.width, ref.height, ref.format, matrix_s='709')
	image = image * int(length)
	image = vstool.FadeIn(image, 2)
	image = vstool.FadeOut(image, 2)
	clip = clip + image

clip = clip[1:]
clip_resize = core.std.AddBorders(clip, smooth.width - clip.width - 1150, smooth.width - clip.width - 162,
                                  ((smooth.height - clip.height) / 2) - 1, ((smooth.height - clip.height) / 2) + 1)
mask = core.std.BlankClip(clip, format=vs.GRAY16, color=255 * 255)
mask = core.std.AddBorders(mask, smooth.width - clip.width - 1150, smooth.width - clip.width - 162,
                           ((smooth.height - clip.height) / 2) - 1, ((smooth.height - clip.height) / 2) + 1)

smooth_new = core.std.MaskedMerge(smooth[616224:647840], clip_resize, mask)

vfr = muvf.VFRSplice([
    final[:122664], smooth[613320:616224], smooth_new, final[129568:final.num_frames - 1],
    core.std.BlankClip(final, length=120)
], 'violet_movie1_timecode.txt')

vfr = core.fmtc.bitdepth(vfr, bits=10, dmode=3)

vfr.set_output()