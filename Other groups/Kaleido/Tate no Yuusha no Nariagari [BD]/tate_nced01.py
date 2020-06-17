"""Tate no Yuusha script"""
__author__ = 'Vardë'

import sys
from typing import NamedTuple
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from vsutil import depth, vs, core, get_y

import vardefunc as vdf
import mvsfunc as mvf
import lvsfunc as lvf

X264 = r'C:\Encode Stuff\x264_tmod_Broadwell_r3000\mcf\x264_x64.exe'

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


JPBD = infos_bd(r'[BDMV][190424][Tate no Yuusha no Nariagari][Vol.1]\TATE_1_2\BDMV\STREAM\00014', 24, -24)
USBD = infos_bd(r'[BDMV] The Rising of the Shield Hero S01 Part 1\[BDMV] Rising_Shield_Hero_S1P1_D1\BDMV\STREAM\00015', 24+30928, 24+33084+1)
X264_ARGS = dict(
    qpfile=JPBD.qpfile, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:-2', me='umh', subme=10, psy_rd='0.95:0.00', merange=32,
    keyint=360, min_keyint=23, rc_lookahead=60, crf=14, qcomp=0.7, aq_mode=3, aq_strength=1.0
)

def do_filter():
    """Vapoursynth filtering"""
    def _perform_filtering_ending(clip: vs.VideoNode, adapt_mask: vs.VideoNode) -> vs.VideoNode:
        luma = get_y(clip)
        denoise_a = mvf.BM3D(luma, 2.25, 1)
        denoise_b = mvf.BM3D(luma, 1.25, 1)
        denoise = core.std.MaskedMerge(denoise_a, denoise_b, adapt_mask)
        grain = core.grain.Add(denoise, 0.3, constant=True)
        return core.std.MaskedMerge(denoise, grain, adapt_mask)

    src = JPBD.src_cut
    src = depth(src, 16)

    avg = core.std.PlaneStats(src)
    adapt_mask = core.adg.Mask(get_y(avg), 28)

    denoise_chroma = core.knlm.KNLMeansCL(src, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
    denoise_luma = _perform_filtering_ending(src, adapt_mask)

    final = vdf.merge_chroma(denoise_luma, denoise_chroma)

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
    mkv.mux(JPBD.output_final)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
