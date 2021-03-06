"""Tate no Yuusha script"""
__author__ = 'Vardë'

import sys
from pathlib import Path
from typing import NamedTuple, List
from functools import partial
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from vsutil import core, vs, depth, get_y

import vardefunc as vdf
import debandshit as dbs
import mvsfunc as mvf
import modfunc as mdf
import lvsfunc as lvf


class InfosBD(NamedTuple):
    path: str
    src: str
    src_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    src_cut: vs.VideoNode
    a_src: str
    a_src_cut: str
    a_enc_cut: str
    name: str
    qpfile: str
    output: str
    chapter: str
    output_final: str

class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    src_clip = lvf.src(path + '.dgi')
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.mka'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = sys.argv[0][:-3]
    qpfile = name + '_qpfile.log'
    output = name + '.264'
    chapter = 'chapters/tate_' + name[-2:] + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end, src_cut, a_src, a_src_cut, a_enc_cut,
                   name, qpfile, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][190724][Tate no Yuusha no Nariagari][Vol.4 Fin]\TATE_4_1\BDMV\STREAM\00010', 0, -24)
JPBD_NCOP = infos_bd(r'[BDMV][190524][Tate no Yuusha no Nariagari][Vol.2]\TATE_2_2\BDMV\STREAM\00012', 24, -24)
USBD = infos_bd(r'[BDMV] The Rising of the Shield Hero S01 Part 2\[BDMV] Rising_Shield_Hero_S1P2_D1\BDMV\STREAM\00058', 24, -24)
X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'
X264_ARGS = dict(
    qpfile=JPBD.qpfile, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-2', me='umh', subme=10, psy_rd='0.95:0.00', merange=32,
    keyint=360, min_keyint=23, rc_lookahead=60, crf=14, qcomp=0.7, aq_mode=3, aq_strength=1.0
)

def do_filter():
    """Vapoursynth filtering"""
    def _sraa(clip: vs.VideoNode, nnargs: dict, eeargs: dict) -> vs.VideoNode:
        def _nnedi3(clip):
            return clip.nnedi3.nnedi3(0, False, **nnargs)
        def _eedi3(clip, sclip):
            return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=sclip)
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        return clip

    def _nnedi3(clip: vs.VideoNode, factor: float, args: dict) -> vs.VideoNode:
        upscale = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
            .std.Transpose().nnedi3.nnedi3(0, True, **args)
        return core.resize.Spline36(
            upscale, clip.width * factor, clip.height * factor,
            src_top=.5, src_left=.5)

    def _line_mask(clip: vs.VideoNode, thr: int) -> vs.VideoNode:
        mask = core.std.Prewitt(clip)
        mask = core.std.Expr(mask, 'x 2 *').std.Median()
        mask = core.std.Expr(mask, f'x {thr} < x x 3 * ?')
        return mask.std.Inflate().std.Deflate()

    def _ssharp(clip: vs.VideoNode, strength: float, width: int, height: int,
                factor: float = 2, b: float = -1, c: float = 6) -> vs.VideoNode:
        source = clip
        sharp = core.resize.Bicubic(clip, clip.width*factor, clip.height*factor, \
            filter_param_a=b, filter_param_b=c).resize.Lanczos(width, height)

        source = core.resize.Spline64(source, sharp.width, sharp.height)

        sharp = core.rgvs.Repair(sharp, source, 13)
        sharp = mvf.LimitFilter(source, sharp, thrc=0.5, elast=6, brighten_thr=0.5, planes=0)


        final = core.std.Expr([sharp, source], f'x {strength} * y 1 {strength} - * +')
        return final

    def to_gray(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        clip = core.std.AssumeFPS(clip, ref)
        return core.resize.Point(clip, format=vs.GRAY16, matrix_s=mvf.GetMatrix(ref))

    def _perform_masks_credit(path: Path) -> List[MaskCredit]:
        return [MaskCredit(lvf.src(str(mask)), int(str(mask.stem).split('_')[2]),
                           int(str(mask.stem).split('_')[3]))
                for mask in path.glob('*')]

    # pylint: disable=unused-argument
    def _diff(n: int, f: vs.VideoFrame, new: vs.VideoNode, adapt: vs.VideoNode) -> vs.VideoNode:
        psa = f.props['PlaneStatsAverage']
        if psa > 0.5:
            clip = new
        elif psa < 0.4:
            clip = adapt
        else:
            weight = (psa - 0.4) * 10
            clip = core.std.Merge(adapt, new, weight)
        return clip


    src = JPBD.src_cut
    src = depth(src, 16)


    denoise = mdf.hybriddenoise_mod(src, 0.55, 2.25)
    diff = core.std.MakeDiff(src, denoise, [0, 1, 2])



    luma = get_y(denoise)

    upscale = _nnedi3(luma, 1.5, dict(nsize=0, nns=3, qual=1, pscrn=1))
    sraa = _sraa(upscale, dict(nsize=0, nns=3, qual=1, pscrn=1),
                 dict(alpha=0.2, beta=0.5, gamma=80, nrad=3, mdis=18))
    sraa = core.rgvs.Repair(sraa, upscale, 3)
    rescale = _ssharp(sraa, 0.55, src.width, src.height, 2)

    artefacts_mask = core.std.BlankClip(rescale, color=(256 << 8) - 1)
    artefacts_mask = vdf.region_mask(artefacts_mask, 2, 2, 2, 2).std.Inflate()
    rescale = core.std.MaskedMerge(luma, rescale, artefacts_mask)

    lineart_mask = _line_mask(luma, 8000)
    antialias = core.std.MaskedMerge(luma, rescale, lineart_mask)

    antialias_merged = vdf.merge_chroma(antialias, denoise)
    antialias_merged = lvf.rfs(antialias_merged, denoise, [(1298, 1415)])


    credit = antialias_merged

    masks_credit_ = _perform_masks_credit(Path('masks_' + JPBD.name[-2:] + '/'))

    for mask in masks_credit_:
        credit = lvf.rfs(credit, core.std.MaskedMerge(credit, denoise, to_gray(mask.mask, src).std.Deflate()),
                         [(mask.start_frame, mask.end_frame)])


    deband_mask = lvf.denoise.detail_mask(credit, brz_a=3000, brz_b=1500)
    deband = dbs.f3kpf(credit, 18, 36, 36)
    deband = core.std.MaskedMerge(deband, credit, deband_mask)



    grain_original = core.std.MergeDiff(deband, diff, [0, 1, 2])
    grain_new = core.neo_f3kdb.Deband(deband, preset='depth', grainy=32, grainc=32)
    avg = core.std.PlaneStats(deband)
    adapt_mask = core.adg.Mask(get_y(avg), 28)
    grain_adapt = core.std.MaskedMerge(grain_new, grain_original, adapt_mask)


    grain = core.std.FrameEval(deband, partial(_diff, new=grain_new, adapt=grain_adapt), avg)


    grain = lvf.rfs(grain, grain_original, [(2282, 2447), (5454, 5529), (23822, 23883)])

    return depth(grain, 10)


def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vdf.gk(JPBD.src_cut, JPBD.qpfile)

    print('\n\n\nVideo encoding')
    vdf.encode(filtered, X264, JPBD.output, **X264_ARGS)

    print('\n\n\nAudio extraction')
    track_01 = USBD.a_src + '_eng.w64'
    track_02 = USBD.a_src + '_jpn.w64'
    eac3to_args = ['eac3to', USBD.src, '3:', track_01, '4:', track_02, '-log=NUL']
    vdf.subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')
    mka = MKVFile()
    mka.add_track(MKVTrack(track_01, 0))
    mka.add_track(MKVTrack(track_02, 0))
    mka.mux(USBD.a_src)

    print('\n\n\nAudio cutting')
    eztrim(USBD.src_clip, [(USBD.frame_start, USBD.frame_end)], USBD.a_src, mkvextract_path='mkvextract')

    print('\n\n\nAudio encoding')
    for i in range(1, len(mka.tracks) + 1):
        qaac_args = ['qaac64', USBD.a_src_cut.format(i), '-V', '127', '--no-delay', '-o', USBD.a_enc_cut.format(i)]
        vdf.subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(JPBD.output, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(USBD.a_enc_cut.format(2), language='jpn', default_track=True))
    mkv.add_track(MKVTrack(USBD.a_enc_cut.format(1), language='eng', default_track=False))
    mkv.chapters(JPBD.chapter, 'jpn')
    mkv.mux(JPBD.output_final)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
