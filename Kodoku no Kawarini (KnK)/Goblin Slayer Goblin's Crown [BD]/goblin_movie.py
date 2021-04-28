"""Goblin Raper script"""
__author__ = 'Vardë'

import sys
import subprocess
from typing import NamedTuple, Dict, Any, Optional
from pathlib import Path
from pymkv import MKVFile, MKVTrack
from acsuite import eztrim

import debandshit as dbs
import muvsfunc as muvf
import vardefunc as vdf
import havsfunc as hvf
import mvsfunc as mvf
import placebo


from vsutil import depth, get_y, get_w, iterate
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

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


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    src_clip = lvf.src(path + '.m2ts')
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'BJS_81630\BDMV\STREAM\00001', 264, -48)
X265 = r'C:\Encode Stuff\x265-3.4+12-geff9_vs2015-AVX2\x265.exe'

def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                   knlm_args: Optional[Dict[str, Any]] = None,
                   bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
    knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
    if knlm_args is not None:
        knargs.update(knlm_args)

    b3args = dict(radius1=1, profile1='fast')
    if bm3d_args is not None:
        b3args.update(bm3d_args)

    luma = get_y(clip)
    luma = mvf.BM3D(luma, sigma, **b3args)
    chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

    return vdf.merge_chroma(luma, chroma)


def _ssim_downsample(clip: vs.VideoNode, width: int, height: int)-> vs.VideoNode:
    clip = muvf.SSIM_downsample(clip, width, height, kernel='Bicubic')
    return depth(clip, 16)

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    h = 720
    w = get_w(h)


    fixedges = lvf.ef(src, [2, 1, 1])
    out = depth(fixedges, 16)



    denoise = hybrid_denoise(out, 0.65, 3)
    denoise_b = hybrid_denoise(out, 0.3, 1)
    denoise_c = hvf.KNLMeansCL(out, 5, 3, 6, 2, device_type='gpu', device_id=0)

    denoise = lvf.rfs(denoise, denoise_b, [(75124, 82725)])
    denoise = vdf.fade_filter(denoise, denoise, denoise_c, 82726, 82790)
    denoise = lvf.rfs(denoise, denoise_c, [(82791, src.num_frames-1)])

    diff = core.std.MakeDiff(out, denoise)
    out = denoise



    luma = get_y(out)

    line_mask = vdf.edge_detect(luma, 'FDOG', 7000, (1, 1))
    line_mask = iterate(line_mask, core.std.Median, 4)


    # Not FSRCNNX this time )^:
    descale = depth(core.descale.Debilinear(depth(luma, 32), w, h), 16)
    upscale = lvf.sraa(descale, 2, 3, src.width, src.height, _ssim_downsample,
                       alpha=0.2, beta=0.6, gamma=400, nrad=2, mdis=15)
    rescaled = core.std.MaskedMerge(luma, upscale, line_mask)
    out = vdf.merge_chroma(rescaled, out)


    # Slight sharp though CAS
    sharp = hvf.LSFmod(out, strength=75, Smode=3, Lmode=2, edgemode=1, edgemaskHQ=True)
    out = sharp



    deband_mask = detail_mask_func(out, brz_a=2000, brz_b=1200)
    deband = dbs.f3kpf(out, 16, 30, 30)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband



    avg = out.std.PlaneStats()
    adg_mask16 = core.adg.Mask(avg, 16)

    grain_hi = core.std.MergeDiff(out, diff)
    grain_hi = core.std.Merge(grain_hi, out, [0, 0.5]) # Half the grain on chroma planes
    grain_hi = core.std.MaskedMerge(out, grain_hi, core.adg.Mask(avg, 6))

    grain_lo = core.neo_f3kdb.Deband(out, preset='depth', grainy=24, keep_tv_range=True)
    grain_lo = core.std.MaskedMerge(grain_lo, grain_hi, core.adg.Mask(avg, 14))

    grain_b = core.std.MaskedMerge(out, out.grain.Add(0.1, constant=True), adg_mask16)

    grain = lvf.rfs(grain_hi, grain_lo, [(0, 16168), (24078, 41084), (63274, 75123)])
    grain = lvf.rfs(grain, grain_b, [(75124, 82725)])


    # Credits
    deband_c = placebo.deband(sharp, 17, 6, 3, 0)
    grain_c = core.std.MaskedMerge(deband_c, deband_c.grain.Add(6.5, constant=True), adg_mask16)
    grain = lvf.rfs(grain, grain_c, [(82791, src.num_frames-1)])
    grain = vdf.fade_filter(grain, grain, grain_c, 82726, 82790)

    out = grain



    return depth(out, 10)


def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
    x265_args = [
        X265, "--y4m", "--frames", f"{clip.num_frames}", "--sar", "1",
        "--colormatrix", "bt709", "--colorprim", "bt709", "--transfer", "bt709", "--range", "limited",
        "--min-luma", str(16<<2), "--max-luma", str(235<<2), "--output-depth", "10",
        "--fps", f"{clip.fps_num}/{clip.fps_den}",
        "-o", JPBD.output, "-",
        # "--frame-threads", "16",
        "--no-sao",
        "--preset", "slower",
        "--crf", "15.5", "--qcomp", "0.80",
        "--bframes", "16",
        "--psy-rd", "2.25", "--psy-rdoq", "1.15",
        "--deblock", "-2:-2",
        "--rc-lookahead", "120",
        "--min-keyint", "23", "--keyint", "360",
        "--aq-mode", "3", "--aq-strength", "0.85"
        ]
    print("Encoder command: ", " ".join(x265_args), "\n")
    process = subprocess.Popen(x265_args, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '3:', JPBD.a_src, '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, JPBD.a_src_cut.format(1))

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac64', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')
    # opus_args = ['ffmpeg', '-i', JPBD.a_src_cut.format(1), '-c:a', 'libopus', '-b:a', '192k', '-y', JPBD.a_enc_cut.format(1)]
    # subprocess.run(opus_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv = MKVFile()
    mkv.add_track(MKVTrack(JPBD.output, language='jpn', default_track=True))
    mkv.add_track(MKVTrack(JPBD.a_enc_cut.format(1), language='jpn', default_track=True))
    mkv.chapters(JPBD.chapter, 'jpn')
    mkv.mux(JPBD.output_final)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
