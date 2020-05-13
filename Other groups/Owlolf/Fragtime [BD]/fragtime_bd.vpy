"""
Fragtime BD
"""

from typing import List, Tuple

from acsuite import eztrim
from mvsfunc import BM3D
from G41Fun import ContraSharpening
from muvsfunc import VFRSplice
from xvs import mvfrc, getY
from lvsfunc import detail_mask
from vapoursynth import FLOAT, YUV, YUV420P16, \
    YUV420P10, VideoNode, core, Format

core.max_cache_size = 16384

path = r"[200513]「フラグタイム」Blu-ray Timeless Edition【初回限定生産】\BD\BDMV\STREAM\00000"
src = core.lsmas.LWLibavSource(path + ".m2ts")

eztrim(src, (48, -24), path[:-5] + "audios.mka", 'fragtime_audio_cut.mka')

src = src[48:-24]


def _resample(clip: VideoNode, out_f: Format):
    return core.resize.Point(clip, format=out_f, dither_type="error_diffusion")

src_a, src_b = [_resample(src, _f) for _f in [core.register_format(YUV, FLOAT, 32, 1, 1), YUV420P16]]



den = BM3D(src_a, [1.25, 1], radius1=1)
contra = ContraSharpening(den, src_a, planes=0)

desc = core.descale.Debicubic(getY(contra), 1500, 844, 1/3, 1/3)

def _nnedi3(clip: VideoNode, width: int, height: int,
            field: Tuple[float, float], shift: Tuple[float, float], **args):
    args = args or dict(nsize=4, nns=4, qual=2)
    return core.resize.Spline16(
        clip.std.Transpose().nnedi3.nnedi3(field[0], True, **args) \
            .std.Transpose().nnedi3.nnedi3(field[1], True, **args),
        width, height, src_top=shift[0], src_left=shift[1]
    )

scaled = _nnedi3(desc, 1920, 1080, (0, 0), (.5, .5))

def _merge_chroma(clips: List[VideoNode]):
    return core.std.ShufflePlanes(clips, [0, 1, 2], YUV)

merged = _merge_chroma([scaled, contra])
out = _resample(merged, YUV420P16)




db_m = detail_mask(out, brz_a=3000, brz_b=1500)

y = getY(out)
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





smooth = mvfrc(src_b[80130:85956], num=60, den=1, preset='slow')

clips_ = [
    src[:642], grain[642:79856], src[79856:80130],
    smooth, src[85956:]
]

vfr = VFRSplice(clips_, 'fragtime_bd_timecode.txt')





final = _resample(vfr, YUV420P10)
final = core.std.Limiter(final, 16, [235 << 2, 240 << 2])

final.set_output(0)
