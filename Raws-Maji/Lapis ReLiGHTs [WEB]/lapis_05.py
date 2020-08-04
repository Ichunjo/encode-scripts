"""Lapis ReLight script"""
__author__ = 'Vardë'

import sys
from pathlib import Path
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from cooldegrain import CoolDegrain
from finedehalo import fine_dehalo
import G41Fun as gf
import kagefunc as kgf
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf


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
AUDIO_SRC = Path(VIDEO_FILES[1]).stem + '.mka'

X264_ARGS = dict(
    threads=27, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='umh', subme=10, psy_rd='0.90:0.00', merange=24,
    keyint=240, min_keyint=23, rc_lookahead=72, crf=15, qcomp=0.7, aq_mode=3, aq_strength=0.85
)

def do_filter():
    """Vapoursynth filtering"""
    def _single_rate_aa(clip):
        nnargs = dict(nsize=0, nns=2, qual=1)
        eeargs = dict(alpha=0.2, beta=0.4, gamma=250, nrad=3, mdis=20)
        clip = core.eedi3m.EEDI3(clip, 0, 0, 0, sclip=core.nnedi3.nnedi3(clip, 0, 0, 0, **nnargs), **eeargs).std.Transpose()
        clip = core.eedi3m.EEDI3(clip, 0, 0, 0, sclip=core.nnedi3.nnedi3(clip, 0, 0, 0, **nnargs), **eeargs).std.Transpose()
        return clip

    # Fun part / Cutting
    src_funi, src_wkn = [depth(c, 16) for c in CLIPS_SRC]
    src_funi, src_wkn = src_funi[744:], src_wkn[0] + src_wkn

    # Dehardsubbing
    # comp = lvf.comparison.stack_compare(src_funi, src_wkn, height=540, make_diff=True)
    src = core.std.MaskedMerge(src_wkn, src_funi, kgf.hardsubmask(src_wkn, src_funi))
    hardsub_rem = core.std.MaskedMerge(src, src_funi, kgf.hardsubmask_fades(src, src_funi, 8, 2000))
    src = lvf.rfs(src, hardsub_rem, [(3368, 3451)])

    # Regular filterchain
    op, ed, eyec = (1200, 3356), (18989, 19108), (31888, 34046)
    opstart = op[0]


    denoise = CoolDegrain(src, tr=1, thsad=36, blksize=8, overlap=4, plane=4)


    h = 846
    w = get_w(h)
    b, c = vdf.get_bicubic_params('mitchell')

    luma = get_y(denoise)



    descale = core.descale.Debicubic(depth(luma, 32), w, h, b, c)

    upscale = vdf.fsrcnnx_upscale(depth(descale, 16), descale.height*2, "Shaders/FSRCNNX_x2_56-16-4-1.glsl", core.resize.Point)
    sraa = _single_rate_aa(upscale)
    rescaled = core.resize.Bicubic(sraa, src.width, src.height, filter_param_a=0, filter_param_b=0)

    dering = core.std.MaskedMerge(rescaled, core.bilateral.Gaussian(rescaled, 0.3), hvf.HQDeringmod(rescaled, incedge=True, show=True))
    rescaled = lvf.rfs(rescaled, dering, [(opstart, opstart+498)])
    merged = vdf.merge_chroma(rescaled, src)
    out = merged

    dehalo = fine_dehalo(out, rx=2.2, darkstr=0, brightstr=1, contra=True, useMtEdge=True)
    out = dehalo



    cred_m = vdf.drm(src, h, b=b, c=c, mthr=80, mode='ellipse')
    credit = lvf.rfs(out, core.std.MaskedMerge(out, src, cred_m), [op])
    credit = lvf.rfs(credit, src, [eyec, ed])
    out = credit


    thr = 7000
    line_mask = gf.EdgeDetect(out, 'FDOG').std.Median().std.Expr(f'x {thr} < x x 3 * ?')

    deband = dbs.f3kpf(out, 17, 36, 36)
    deband_b = dbs.f3kbilateral(out, 20, 64, 64)
    deband = lvf.rfs(deband, deband_b, [(14775, 14846)])
    deband = core.std.MaskedMerge(deband, out, line_mask)
    out = deband

    dedimm = gf.Tweak(out, sat=1.1, cont=1.1)
    dedimm = lvf.rfs(out, dedimm, [(12957, 13061)])
    out = dedimm


    grain = core.grain.Add(out, 0.1)
    out = grain

    return depth(out, 10)


def do_encode(filtered):
    """Compression with x264"""
    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, OUTPUT, **X264_ARGS, frames=filtered.num_frames)

    print('\n\n\nAudio extracting')
    mka = MKVFile()
    mka.add_track(MKVTrack(VIDEO_FILES[1], 1))
    mka.mux(AUDIO_SRC)

    print('\n\n\nAudio cutting')
    eztrim(CLIPS_SRC[1], (0, 1), AUDIO_SRC, outfile=NAME + '_a.mka')
    eztrim(CLIPS_SRC[1], (0, 0), AUDIO_SRC, outfile=NAME + '_b.mka')

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', NAME + '.mkv',
                '--language', '0:jpn', OUTPUT,
                '--language', '0:jpn', NAME + '_a' + '.mka', '+', NAME + '_b' + '.mka']
    vdf.subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
