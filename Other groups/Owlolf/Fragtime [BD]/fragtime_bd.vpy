"""
Fragtime BD
"""

from typing import List, Tuple

from acsuite import eztrim
import mvsfunc as mvf
import G41Fun as gf
from muvsfunc import VFRSplice
import xvs
from lvsfunc.denoise import detail_mask
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

path = r"[200513]「フラグタイム」Blu-ray Timeless Edition【初回限定生産】\BD\BDMV\STREAM\00000"
src = core.lsmas.LWLibavSource(path + ".m2ts")

# Compensating the audio desync
eztrim(src, (46, -24), path[:-5] + "audios.mka", 'fragtime_audio_cut.mka')

src = src[48:-24]


src_a, src_b = [core.resize.Point(src, format=_f, dither_type="error_diffusion")
                for _f in [core.register_format(vs.YUV, vs.FLOAT, 32, 1, 1), vs.YUV420P16]]



den = mvf.BM3D(src_a, [1.25, 1], radius1=1)
contra = gf.ContraSharpening(den, src_a, planes=0)

desc = core.descale.Debicubic(xvs.getY(contra), 1500, 844, 1/3, 1/3)
desc = core.resize.Point(desc, format=vs.GRAY16)

def _nnedi3(clip: vs.VideoNode, width: int, height: int,
            field: Tuple[float, float], shift: Tuple[float, float], **args):
    args = args or dict(nsize=1, nns=4, qual=2, pscrn=2)
    return core.resize.Spline64(
        clip.std.Transpose().nnedi3.nnedi3(field[0], True, **args) \
            .std.Transpose().nnedi3.nnedi3(field[1], True, **args),
        width, height, src_top=shift[0], src_left=shift[1]
    )

aa = _nnedi3(desc, 1500, 844, (0, 0), (.5, .5))


blank = core.std.BlankClip(aa, format=vs.GRAY16, color=[128 << 8])
descaled = core.std.ShufflePlanes([aa, blank, blank], [0, 0, 0], vs.YUV)

double = core.placebo.Shader(descaled, "Shaders/FSRCNNX_x2_56-16-4-1.glsl",
                             descaled.width * 2, descaled.height * 2)

rescale = core.resize.Spline64(xvs.getY(double), 1920, 1080)


merged = core.std.ShufflePlanes([rescale, contra.resize.Point(format=vs.YUV420P16)],
                                [0, 1, 2], vs.YUV)
out = core.resize.Point(merged, format=vs.YUV420P16)


db_m = detail_mask(out, brz_a=3000, brz_b=1500)

y = xvs.getY(out)
adapt_m = core.adg.Mask(y.std.PlaneStats(), 8)

db = core.std.MaskedMerge(
    core.neo_f3kdb.Deband(out, 17, 30, 24, 24, 24, 0, sample_mode=4),
    core.neo_f3kdb.Deband(out, 17, 48, 42, 42, 24, 0, sample_mode=4),
    adapt_m
)

db = core.std.MaskedMerge(db, out, db_m)

grain = core.std.MaskedMerge(
    db, core.grain.Add(db, 0.25, constant=True),
    core.std.BoxBlur(adapt_m, 0, 1, 1, 1, 1), 0
)





smooth = xvs.mvfrc(src_b[80130:85956], num=60000, den=1001, preset='slow')

clips_ = [
    src[:642], grain[642:79856], src[79856:80130], smooth, src[85956:]
]

vfr = VFRSplice(clips_, 'fragtime_bd_timecode.txt')



final = core.resize.Point(vfr, format=vs.YUV420P10)
final = core.std.Limiter(final, 16 << 2, [235 << 2, 240 << 2])

final.set_output(0)
