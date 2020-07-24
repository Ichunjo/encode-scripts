"""Shinsekai Yori script"""
__author__ = 'Vardë'

import sys
import subprocess
from functools import partial
from typing import NamedTuple
from pathlib import Path
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
from muvsfunc import SSIM_downsample
import debandshit as dbs
import vardefunc as vdf
import kagefunc as kgf
import placebo
import shinfunc as shf
import shinyori_ed01


from vsutil import depth, get_y, get_w
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.num_threads = 12

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
    output: str
    chapter: str
    output_final: str


def infos_bd(path, frame_start, frame_end)-> InfosBD:
    src = path + '.m2ts'
    src_clip = lvf.src(path + '.m2ts')
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.mka'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][Shinsekai Yori]\[BDMV][121130][Shinsekai Yori][Vol.01]\BDMV\STREAM\00000', 0, -24)
X265 = r'C:\Encode Stuff\x265-3.4+12-geff9_vs2015-AVX2\x265.exe'

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 32)
    ed = (30089, 32247)

    denoise = kgf.hybriddenoise(src, 0.45, 2)
    out = denoise

    h = 720
    w = get_w(h)
    b, c = vdf.get_bicubic_params('mitchell')


    luma = get_y(out)
    line_mask = shf.edge_mask_simple(luma, 'FDOG', 0.08, (1, 1))


    descale = core.descale.Debicubic(luma, w, h, b, c)
    upscale = shf.fsrcnnx_upscale(descale, src.height, 'shaders/FSRCNNX_x2_56-16-4-1.glsl',
                                  partial(SSIM_downsample, kernel='Bicubic'))
    rescale = core.std.MaskedMerge(luma, upscale, line_mask)
    merged = vdf.merge_chroma(rescale, denoise)
    out = depth(merged, 16)



    mask = shf.detail_mask(out, (10000, 4000), (12000, 3500), [(2, 2), (2, 2)], sigma=[50, 250, 400], upper_thr=0.005)
    deband = dbs.f3kpf(out, 17, 42, 48, thrc=0.4)
    deband = core.std.MaskedMerge(deband, out, mask)

    deband_b = placebo.deband(out, 27, 8, 3, 0)
    deband = lvf.rfs(deband, deband_b, [(3404, 3450)])

    deband_c = shf.deband_stonks(out, 20, 8, 3, shf.edge_mask_simple(out, 'prewitt', 2500, (8, 1)))
    deband = lvf.rfs(deband, deband_c, [(5642, 5784), (6222, 6479), (7798, 8073), (8133, 8256), (9699, 9817)])

    deband_d = placebo.deband(out, 17, 7.5, 1, 0)
    deband_d = core.std.MaskedMerge(deband_d, out, mask)
    deband = lvf.rfs(deband, deband_d, [(8074, 8132), (8711, 8766), (12267, 12433), (28468, 28507)])

    grain = core.neo_f3kdb.Deband(deband, preset='depth', grainy=24, grainc=24)
    out = grain


    grain = adptvgrnMod(out, 0.3, size=4/3, sharp=55, luma_scaling=14, grain_chroma=False)
    out = grain


    ending = shinyori_ed01.filtering(src, *ed)
    final = lvf.rfs(out, ending, [ed])

    return depth(final, 10)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x264"""
    print('\n\n\nVideo encoding')
    x265_args = [
        X265, "--y4m", "--frames", f"{clip.num_frames}", "--sar", "1", "--output-depth", "10",
        "--colormatrix", "bt709", "--colorprim", "bt709", "--transfer", "bt709", "--range", "limited",
        "--min-luma", str(16<<2), "--max-luma", str(235<<2),
        "--fps", f"{clip.fps_num}/{clip.fps_den}",
        "-o", JPBD.output, "-",
        "--frame-threads", "16",
        "--no-sao", "--fades",
        "--preset", "slower",
        "--crf", "14.5", "--qcomp", "0.72",
        "--bframes", "16",
        "--psy-rd", "2.0", "--psy-rdoq", "1.0",
        "--deblock", "-1:-1",
        "--rc-lookahead", "96",
        "--min-keyint", "23", "--keyint", "360",
        "--aq-mode", "3", "--aq-strength", "1.0"
        ]
    print("Encoder command: ", " ".join(x265_args), "\n")
    process = subprocess.Popen(x265_args, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()

    print('\n\n\nAudio extraction')
    mka = MKVFile()
    mka.add_track(MKVTrack(JPBD.src, 1))
    mka.mux(JPBD.a_src)

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, mkvextract_path='mkvextract')

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac64', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(JPBD.output, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(JPBD.a_enc_cut.format(1), language='jpn', default_track=True))
    mkv.chapters(JPBD.chapter, 'jpn')
    mkv.mux(JPBD.output_final)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
