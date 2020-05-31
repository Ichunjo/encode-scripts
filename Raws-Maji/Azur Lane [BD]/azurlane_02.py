"""Azur Lane script"""
__author__ = 'Vardë'

import sys
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

PATH = r'アズールレーン\[191204]アニメ『アズールレーン』VOLUME 1\BD1\BDMV\STREAM\00001'
SRC = PATH + '.m2ts'
SRC_CLIP = lvf.src(SRC)

FRAME_START, FRAME_END = 24, -24
SRC_CUT = SRC_CLIP[FRAME_START:FRAME_END]

OPSTART, OPEND = 1630, 3788
EDSTART, EDEND = 32128, SRC_CUT.num_frames - 1

A_SRC = PATH + '.wav'
A_SRC_CUT = PATH + '_cut_track_1.wav'
A_ENC_CUT = PATH + '.m4a'
NAME = sys.argv[0][:-3]
QPFILE = NAME + '_qpfile.log'
OUTPUT = NAME + '.264'
CHAPTER = 'アズールレーン/chapter' + NAME[-2:] + '.txt'
OUTPUT_FINAL = NAME + '.mkv'

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=QPFILE, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='umh', subme=10, psy_rd='0.90:0.00', merange=24,
    keyint=360, min_keyint=1, rc_lookahead=60, crf=15, qcomp=0.7, aq_mode=3, aq_strength=0.9
)

def do_filter():
    """Vapoursynth filtering"""
    def _nneedi3_clamp(clip: vs.VideoNode, strength: int = 1):
        bits = clip.format.bits_per_sample - 8
        thr = strength * (1 >> bits)

        luma = get_y(clip)

        strong = TAAmbk(luma, aatype='Eedi3', alpha=0.4, beta=0.4)
        weak = TAAmbk(luma, aatype='Nnedi3')
        expr = 'x z - y z - * 0 < y x y {0} + min y {0} - max ?'.format(thr)

        clip_aa = core.std.Expr([strong, weak, luma], expr)
        return core.std.ShufflePlanes([clip_aa, clip], [0, 1, 2], vs.YUV)

    src = SRC_CUT

    interpolate = core.resize.Bicubic(src, src_left=3)
    f_1 = 1006
    src = src[:EDSTART+f_1] + interpolate[EDSTART+f_1] + src[EDSTART+f_1:-1]

    src = depth(src, 16)

    blur = core.bilateral.Gaussian(src, [0.45, 0])
    den = CoolDegrain(blur, tr=1, thsad=60, blksize=8, overlap=4, plane=4)



    dering = core.std.MaskedMerge(den, hvf.YAHR(den, 2, 32), muvf.AnimeMask(den, 0.2), 0)




    aa_a = core.std.MaskedMerge(dering, _nneedi3_clamp(dering), TAAmbk(dering, mtype=2, showmask=1))
    aa_b = TAAmbk(src, aatype='Nnedi3', mtype=1, nns=4, qual=2, nsize=6)
    aa = lvf.rfs(aa_a, aa_b, [(OPSTART, OPEND), (EDSTART, EDEND)])




    preden = core.knlm.KNLMeansCL(aa, a=2, h=2, d=0, device_type='gpu', channels='Y')
    diff = core.std.MakeDiff(aa, preden, 0)
    db_m = lvf.denoise.detail_mask(aa.std.Median(), brz_a=3000, brz_b=1500)

    db_a = dbs.f3kpf(aa, 17)
    db_b = core.placebo.Deband(preden, radius=17, threshold=5.5, iterations=1, grain=4, planes=1|2|4)
    db_b = core.std.MergeDiff(db_b, diff, 0)
    db = lvf.rfs(db_a, db_b, [(9729, 9845), (13652, 14048)])
    db = core.std.MaskedMerge(db, aa, db_m)

    grain = mdf.adptvgrnMod_mod(db, 0.2, size=1.25, sharp=60, luma_scaling=8)

    final = depth(grain, 10)

    return final, src


def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vrf.gk(SRC_CUT, QPFILE)

    print('\n\n\nVideo encode')
    vrf.encode(filtered, X264, OUTPUT, **X264_ARGS)

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', SRC, '2:', A_SRC, '-log=NUL']
    vrf.subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cut')
    eztrim(SRC_CLIP, (FRAME_START, FRAME_END), A_SRC, mkvextract_path='mkvextract')

    print('\n\n\nAudio encode')
    qaac_args = ['qaac64', A_SRC_CUT, '-V', '127', '--no-delay', '-o', A_ENC_CUT]
    vrf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal mux')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(OUTPUT, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(A_ENC_CUT, language='jpn', default_track=True))
    mkv.chapters(CHAPTER, 'jpn')
    mkv.mux(OUTPUT_FINAL)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED[0])
