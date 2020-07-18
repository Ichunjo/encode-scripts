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

    def _nnedi3(clip: vs.VideoNode, factor: float, args: dict) -> vs.VideoNode:
        upscale = clip.std.Transpose().nnedi3cl.NNEDI3CL(0, True, **args) \
            .std.Transpose().nnedi3cl.NNEDI3CL(0, True, **args)
        sraa = _sraa(upscale, dict(nsize=3, nns=3, qual=1, pscrn=1),
                dict(alpha=0.3, beta=0.4, gamma=500, nrad=3, mdis=18))
        return core.resize.Bicubic(sraa, clip.width*factor, clip.height*factor,
                                   src_top=.5, src_left=.5, filter_param_a=0, filter_param_b=0)

    def _sraa(clip: vs.VideoNode, nnargs: dict, eeargs: dict) -> vs.VideoNode:
        def _nnedi3(clip):
            return clip.nnedi3.nnedi3(0, False, **nnargs)
        def _eedi3(clip, sclip):
            return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=sclip)
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        return clip

    # Fun part / Cutting
    src_funi, src_wkn = [depth(c, 16) for c in CLIPS_SRC]
    src_funi, src_wkn = src_funi[744:], src_wkn[0] + src_wkn

    # Dehardsubbing
    # comp = lvf.comparison.stack_compare(src_funi, src_wkn, height=720, make_diff=True)
    src = core.std.MaskedMerge(src_wkn, src_funi, kgf.hardsubmask(src_wkn, src_funi))
    hardsub_rem = core.std.MaskedMerge(src, src_funi, kgf.hardsubmask_fades(src, src_funi, 8, 2000))
    src = lvf.rfs(src, hardsub_rem, [(3796, 3892)])

    # Regular filterchain
    op, ed, eyec = (1630, 3788), (31888, 34046), (16400, 16519)
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
    credit = lvf.rfs(credit, src, [eyec, ed])
    out = credit

    deband = dbs.f3kpf(out, 17, 36, 36)
    deband = core.std.MaskedMerge(deband, out, line_mask)

    dedimm = gf.Tweak(deband, sat=1.15, cont=1.15)
    dedimm = lvf.rfs(deband, dedimm, [(21093, 21211)])

    grain = core.grain.Add(dedimm, 0.2)

    return depth(grain, 10)


def do_encode(filtered):
    """Compression with x264"""
    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, OUTPUT, True, **X264_ARGS, frames=filtered.num_frames)

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
