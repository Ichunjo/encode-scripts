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
import placebo
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

JPBD = infos_bd(r'[BDMV][190424][Tate no Yuusha no Nariagari][Vol.1]\TATE_1_2\BDMV\STREAM\00012', 0, -24)
JPBD_NCOP = infos_bd(r'[BDMV][190424][Tate no Yuusha no Nariagari][Vol.1]\TATE_1_2\BDMV\STREAM\00013', 24, -24)
USBD = infos_bd(r'[BDMV] The Rising of the Shield Hero S01 Part 1\[BDMV] Rising_Shield_Hero_S1P1_D2\BDMV\STREAM\00014', 24, -24)
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

    def _w2x(clip: vs.VideoNode) -> vs.VideoNode:
        waifu2x = core.w2xc.Waifu2x(mvf.ToRGB(clip, depth=32), noise=2, scale=2) \
            .resize.Bicubic(clip.width, clip.height)
        return mvf.ToYUV(waifu2x, css='420', depth=16)

    def _perform_filtering_ending(clip: vs.VideoNode, adapt_mask: vs.VideoNode) -> vs.VideoNode:
        luma = get_y(clip)
        denoise_a = mvf.BM3D(luma, 2.25, 1)
        denoise_b = mvf.BM3D(luma, 1.25, 1)
        denoise = core.std.MaskedMerge(denoise_a, denoise_b, adapt_mask)
        grain = core.grain.Add(denoise, 0.3, constant=True)
        return core.std.MaskedMerge(denoise, grain, adapt_mask)

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


    opstart, opend = 0, 2157
    edstart, edend = 31240, 33396
    src = JPBD.src_cut
    src += src[-1]
    src = depth(src, 16)
    src = core.std.FreezeFrames(src, opstart+2132, opend, opstart+2132)

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


    src_c, src_ncop = [c.knlm.KNLMeansCL(a=6, h=20, d=0, device_type='gpu')
                       for c in [src, JPBD_NCOP.src_cut[:opend-opstart+1]]]
    credit_mask = vdf.dcm(src, src_c[opstart:opend+1], src_ncop, opstart, opend, 2, 2).std.Deflate()
    credit = core.std.MaskedMerge(antialias_merged, denoise, credit_mask)

    masks_credit_ = _perform_masks_credit(Path('masks_' + JPBD.name[-2:] + '/'))

    for mask in masks_credit_:
        credit = lvf.rfs(credit, core.std.MaskedMerge(credit, denoise, to_gray(mask.mask, src).std.Deflate()),
                         [(mask.start_frame, mask.end_frame)])


    deband_mask = detail_mask_func(credit, brz_a=3000, brz_b=1500)
    deband_a = dbs.f3kpf(credit, 18, 36, 36)
    deband_b = dbs.f3kpf(credit, 18, 42, 42)
    deband_c = placebo.Deband(credit, radius=16, threshold=4, iterations=1, grain=0)
    deband_d = placebo.Deband(deband_b, radius=20, threshold=5, iterations=1, grain=0)
    deband_e = dbs.f3kpf(credit, 16, 18, 18)
    deband = lvf.rfs(deband_a, deband_b, [(opstart, opstart+146)])
    deband = lvf.rfs(deband, deband_c, [(opstart+1225, opstart+1238)])
    deband = lvf.rfs(deband, deband_d, [(opstart+970, opstart+984), (28452, 28508)])
    deband = lvf.rfs(deband, deband_e, [(22217, 22359)])
    deband = core.std.MaskedMerge(deband, credit, deband_mask)


    grain_original = core.std.MergeDiff(deband, diff, [0, 1, 2])
    grain_new = core.neo_f3kdb.Deband(deband, preset='depth', grainy=32, grainc=32)
    avg = core.std.PlaneStats(deband)
    adapt_mask = core.adg.Mask(get_y(avg), 28)
    grain_adapt = core.std.MaskedMerge(grain_new, grain_original, adapt_mask)


    grain = core.std.FrameEval(deband, partial(_diff, new=grain_new, adapt=grain_adapt), avg)

    grain = lvf.rfs(grain, grain_new, [(opstart+147, opstart+496), (opstart+575, opstart+644),
                                       (opstart+702, opstart+969), (opstart+1076, opstart+1117),
                                       (opstart+1428, opstart+1461), (opstart+1859, opstart+2035)])
    grain = lvf.rfs(grain, grain_original, [(33781, 33924)])

    w2x = _w2x(denoise).grain.Add(1, 0.5, constant=True)
    w2x = lvf.rfs(grain, w2x, [(opstart+1211, opstart+1224)])

    ending = _perform_filtering_ending(src, adapt_mask)
    ending = vdf.merge_chroma(ending, denoise)
    final = lvf.rfs(w2x, ending, [(edstart, edend)])

    return depth(final, 10)


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
    eztrim(USBD.src_clip, (USBD.frame_start, USBD.frame_end), USBD.a_src, mkvextract_path='mkvextract')

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
