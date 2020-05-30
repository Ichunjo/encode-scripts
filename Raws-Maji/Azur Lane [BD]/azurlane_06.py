"""
Azur Lane script
"""
from pymkv import MKVFile, MKVTrack


from vsutil import get_y, depth, core, vs
from acsuite import eztrim
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain

import vardefunc as vrf
import debandshit as dbs
import modfunc as mdf
import havsfunc as hvf
import muvsfunc as muvf
import lvsfunc as lvf

core.max_cache_size = 1024 * 16

PATH = r"アズールレーン\[200219]アニメ『アズールレーン』VOLUME 3\BD3\BDMV\STREAM\00001"
SRC = lvf.src(PATH + ".m2ts")

FRAME_START, FRAME_END = 24, -23
SRC_C = SRC[FRAME_START:FRAME_END]

OPSTART, OPEND = 888, 3044
EDSTART, EDEND = 32129, SRC_C.num_frames - 1


NAME = vrf.Path(__file__).stem
A_SRC = PATH + '.mka'
A_SRC_CUT = PATH + '_cut_track_1.wav'
A_ENC_CUT = PATH + '.m4a'
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
CHAPTER = 'アズールレーン/CHAPTER' + NAME[-2:] + '.txt'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r"C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe"
X264_ARGS = dict(
    QPFILE=QPFILE, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct="auto", deblock="-1:-1", me="umh", subme=10, psy_rd="0.90:0.00", merange=24,
    keyint=360, min_keyint=1, rc_lookahead=60, crf=15, qcomp=0.7, aq_mode=3, aq_strength=0.9
)

def do_filter():
    def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1):
        bits = clip.format.bits_per_sample - 8
        thr = strength * (1 >> bits)

        y = get_y(clip)

        strong = TAAmbk(y, aatype='Eedi3', alpha=0.4, beta=0.4)
        weak = TAAmbk(y, aatype='Nnedi3')
        expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

        clip_aa = core.std.Expr([strong, weak, y], expr)
        return core.std.ShufflePlanes([clip_aa, clip], [0, 1, 2], vs.YUV)

    src = SRC_C



    interpolate = core.resize.Bicubic(src, src_left=3)
    src = src[:EDSTART+1005] + interpolate[EDSTART+1005] + src[EDSTART+1005:-1]

    src = depth(src, 16)

    blur = core.bilateral.Gaussian(src, [0.45, 0])
    den = CoolDegrain(blur, tr=1, thsad=60, blksize=8, overlap=4, plane=4)



    dering = core.std.MaskedMerge(den, hvf.YAHR(den, 2, 32), muvf.AnimeMask(den, 0.2), 0)




    aa_a = core.std.MaskedMerge(dering, _nneedi3_clamp(dering), TAAmbk(dering, mtype=2, showmask=1))
    aa_b = TAAmbk(src, aatype='Nnedi3', mtype=1, nns=4, qual=2, nsize=6)
    aa = lvf.rfs(aa_a, aa_b, [(OPSTART, OPEND), (EDSTART, EDEND)])





    db_m = lvf.denoise.detail_mask(aa.std.Median(), brz_a=3000, brz_b=1500)


    db_a = dbs.f3kpf(aa, 17)
    db = core.std.MaskedMerge(db_a, aa, db_m)

    grain = mdf.adptvgrnMod_mod(db, 0.2, size=1.25, sharp=60, luma_scaling=8)

    final = depth(grain, 10)

    return final
    # final.set_output(0)
    # src.set_output(1)


def do_encode(filtered):
    eztrim(SRC, (FRAME_START, FRAME_END), A_SRC, mkvextract_path="mkvextract")
    qaac_args = ['qaac64', A_SRC_CUT, '-V', '127', '--no-delay', '-o', A_ENC_CUT]
    vrf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')


    vrf.gk(SRC_C, QPFILE)


    vrf.encode(filtered, X264, OUTPUT, **X264_ARGS)


    mkv = MKVFile()
    mkv.add_track(MKVTrack(OUTPUT, language="jpn", default_track=True))
    mkv.add_track(MKVTrack(A_ENC_CUT, language="jpn", default_track=True))
    mkv.chapters(CHAPTER, "jpn")
    mkv.mux(OUTPUT_FINAL)


if __name__ == "__main__":
    FILTERED = do_filter()
    do_encode(FILTERED)
