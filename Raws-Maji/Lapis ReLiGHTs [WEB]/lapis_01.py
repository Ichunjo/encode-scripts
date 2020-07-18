"""Lapis ReLight script"""
__author__ = 'Vardë'

import sys
from pathlib import Path
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

import G41Fun as gf
import kagefunc as kgf
import debandshit as dbs
import vardefunc as vdf

from vsutil import depth, get_y, get_w
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core


X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3007\mcf\x264_x64.exe'

NAME = Path(sys.argv[0]).stem
OUTPUT = NAME + '.264'

NUM = NAME[-2:]
VIDEO_FILES = ['FUNI/[HorribleSubs] Lapis ReLiGHTs - {} [1080p].mkv'.format(NUM),
               'WKN/Lapis ReLIGHTs E{} [1080p][AAC][JapDub][GerSub][Web-DL].mkv'.format(NUM)]
CLIPS_SRC = [lvf.src(c) for c in VIDEO_FILES]
AUDIO_SRC = Path(VIDEO_FILES[1]).stem + 'mka'

X264_ARGS = dict(
    threads=27, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='umh', subme=10, psy_rd='0.90:0.00', merange=24,
    keyint=240, min_keyint=23, rc_lookahead=96, crf=15, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_filter():
    """Vapoursynth filtering"""
    def _nnedi3(clip: vs.VideoNode, factor: float, args: dict)-> vs.VideoNode:
        upscale = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
            .std.Transpose().nnedi3.nnedi3(0, True, **args)
        sraa = _sraa(upscale, dict(nsize=3, nns=3, qual=1, pscrn=1))
        return core.resize.Bicubic(sraa, clip.width*factor, clip.height*factor,
                                   src_top=.5, src_left=.5, filter_param_a=0, filter_param_b=0.25)

    # My GPU’s dying on it
    def _sraa(clip: vs.VideoNode, nnargs: dict)-> vs.VideoNode:
        clip = clip.nnedi3cl.NNEDI3CL(0, False, **nnargs).std.Transpose()
        clip = clip.nnedi3cl.NNEDI3CL(0, False, **nnargs).std.Transpose()
        return clip

    # Fun part / Cutting
    src_funi, src_wkn = [depth(c, 16) for c in CLIPS_SRC]
    src_funi, src_wkn = src_funi[744:], src_wkn[0] + src_wkn

    # Dehardsubbing
    # comp = lvf.comparison.stack_compare(src_funi, src_wkn, make_diff=True)
    src = core.std.MaskedMerge(src_wkn, src_funi, kgf.hardsubmask(src_wkn, src_funi))
    hardsub_rem = core.std.MaskedMerge(src, src_funi, kgf.hardsubmask_fades(src, src_funi, 8, 2000))
    src = lvf.rfs(src, hardsub_rem, [(3917, 4024)])


    # Regular filterchain
    op, ed = (17238, 19468), (31889, 34045)
    h = 846
    w = get_w(h)
    b, c = vdf.get_bicubic_params('robidoux')

    luma = get_y(src)

    thr = 8000
    line_mask = gf.EdgeDetect(luma, 'FDOG').std.Median().std.Expr(f'x {thr} < x x 3 * ?')

    descale = core.descale.Debicubic(depth(luma, 32), w, h, b, c)
    rescaled = _nnedi3(depth(descale, 16), src.height/h, dict(nsize=0, nns=4, qual=2, pscrn=2))
    rescaled = core.std.MaskedMerge(luma, rescaled, line_mask)
    merged = vdf.merge_chroma(rescaled, src)
    out = merged

    cred_m = vdf.drm(src, h, b=b, c=c, mthr=80, mode='ellipse')
    credit = lvf.rfs(out, core.std.MaskedMerge(out, src, cred_m), [op])
    credit = lvf.rfs(credit, src, [ed])
    out = credit

    deband = dbs.f3kpf(out, 17, 36, 36)
    deband = core.std.MaskedMerge(deband, out, line_mask)
    grain = core.grain.Add(deband, 0.25)

    return depth(grain, 10)


def do_encode(filtered):
    """Compression with x264"""
    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, OUTPUT, **X264_ARGS, frames=filtered.num_frame)

    print('\n\n\nAudio extracting')
    mka = MKVFile()
    mka.add_track(MKVTrack(VIDEO_FILES[1], 1))
    mka.mux(AUDIO_SRC)

    print('\n\n\nAudio cutting')
    eztrim(CLIPS_SRC[1], (0, 1), AUDIO_SRC, outfile=NAME + '_a', ffmpeg_path='')
    eztrim(CLIPS_SRC[1], (0, 0), AUDIO_SRC, outfile=NAME + '_b', ffmpeg_path='')

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', NAME + '.mkv',
                '--language', '0:jpn', OUTPUT,
                '--language', '0:jpn', NAME + '_a' + '.mka', '+', NAME + '_b' + '.mka']
    vdf.subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
